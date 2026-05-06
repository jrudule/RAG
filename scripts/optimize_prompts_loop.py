"""
Katram jautājumam:
  1. Baseline ģenerēšana
  2. Kļūdu analīze
  3. Uzvednes labošana
  4. Candidate ģenerēšana un salīdzinājums
  5. Ja candidate > baseline, tad uzvedni saglabā prompts.yaml / questions.yaml

Lietojums: python -m scripts.optimize_prompts_loop --questions 39 39.20
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from tqdm import tqdm

from scripts.extractmd import Extractor
from scripts.gen_results import gen_results
from scripts.utilities import (
    get_answers,
    get_config_data,
    get_ini_files,
    get_procurement_content,
    get_prompt_dict,
    get_questions,
    get_supplementary_info,
)
from scripts.vectorindex import QnAEngine

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_CONF = {
    'embeddingmodel': 'BAAI/bge-m3',
    'chunk_size': 1536,
    'chunk_overlap': 0,
    'top_similar': 5,
    'n4rerank': 0,
    'use_similar_chunks': True,
    'prevnext': True,
}

REPORT_COLS = ['Iepirkuma ID', 'Nr', 'Atbilde', 'Sagaidāmā atbilde', 'Pamatojums', 'Uzvedne']


def find_question_data(q_dict, nr):
    for q in q_dict:
        if str(q['nr']) == str(nr):
            return q
        for sq in q.get('questions', []):
            if str(sq['nr']) == str(nr):
                return sq
    return None


def compute_accuracy(csv_path, q_nr):
    if not Path(csv_path).exists():
        return None, 0
    df = pd.read_csv(csv_path, dtype={'Nr': str})
    df['Atbilde'] = df['Atbilde'].astype(str).str.strip().str.lower()
    df['Sagaidama_atbilde'] = df['Sagaidāmā atbilde'].astype(str).str.strip().str.lower()
    df = df[(df['Nr'].astype(str) == str(q_nr)) & (df['Sagaidama_atbilde'] != '?')]
    if len(df) == 0:
        return None, 0
    correct = (df['Atbilde'] == df['Sagaidama_atbilde']).sum()
    return round(correct / len(df) * 100, 2), len(df)


def questions_to_process_for(target_q):
    base = str(target_q).split('.')[0]
    return [base, str(target_q)] if base != str(target_q) else [base]


async def run_report(
    out_dir, out_csv, prompt_dict, ini_files, config_dir, procurement_file_dir,
    answer_file_dir, question_dictionary, supplementary_info, questions_to_process,
    embedding, llm, extractor, label=''
):
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    print(f'Sāk ģenerēšanu: {label}')
    all_rows = []

    for file in tqdm(sorted(ini_files), desc='Config files', unit='file'):
        configfile = config_dir / f'{file}.ini'
        tqdm.write(f'Apstrada: {configfile}')

        _, proc_file, agr_file, ans_file = get_config_data(
            configfile, procurement_file_dir, answer_file_dir
        )
        answer_dict = get_answers(ans_file)
        content = get_procurement_content(extractor, proc_file, agr_file)

        engine = QnAEngine(embedding, llm)
        await engine.createIndex(
            content, 'Procurement',
            chunk_size=EMBEDDING_CONF['chunk_size'],
            chunk_overlap=EMBEDDING_CONF['chunk_overlap'],
        )

        rows = gen_results(
            engine, configfile, EMBEDDING_CONF,
            question_dictionary, answer_dict, prompt_dict,
            supplementary_info, questions_to_process,
        )
        for r in rows:
            r.insert(0, file)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=REPORT_COLS)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    return df


def build_suggestion_request(cur_prompt, q_text, failures_df):
    fail_cols = ['Iepirkuma ID', 'Sagaidāmā atbilde', 'Atbilde', 'Pamatojums']
    failures_str = failures_df[fail_cols].to_string(index=False)
    return (
        'Tu esi eksperts uzvedņu inženierijā. Palīdzi uzlabot LLM uzvedni '
        'iepirkuma dokumentācijas pārbaudes sistēmai.\n\n'
        'Pašreizējā uzvedne:\n' + cur_prompt + '\n\n'
        'Jautājums, kas LLM tiek dots kopā ar uzvedni:\n' + q_text + '\n\n'
        'Nepareizās atbildes (Iepirkuma ID | Sagaidāmā atbilde | LLM atbilde | LLM skaidrojums):\n'
        + failures_str + '\n\n'
        'Uzdevums:\n'
        '1. Analizē, kāpēc LLM kļūdās katrā no gadījumiem.\n'
        '2. Iesaki konkrētas izmaiņas uzvednes tekstā.\n'
        '3. Ja vajadzīgs, pievieno skaidrus nosacījumus "ja", "nē" un "n/a" gadījumiem.\n'
        '4. Atbildi latviski.\n'
        '5. Gatavoto uzvednes tekstu ievieto tieši starp tagiem <suggested_prompt> un </suggested_prompt>'
    )


def extract_failures(csv_path, q_nr):
    df = pd.read_csv(csv_path, dtype={'Nr': str})
    df['Atbilde'] = df['Atbilde'].astype(str).str.strip().str.lower()
    df['Sagaidāmā atbilde'] = df['Sagaidāmā atbilde'].astype(str).str.strip().str.lower()
    df = df[
        (df['Nr'].astype(str) == str(q_nr))
        & (df['Sagaidāmā atbilde'] != '?')
    ]
    return df[df['Atbilde'] != df['Sagaidāmā atbilde']]


def count_usage_in_file(q_list, pid, key):
    count = 0
    for q in q_list:
        if q.get(key) == pid:
            count += 1
        count += count_usage_in_file(q.get('questions', []), pid, key)
    return count


def update_q_prompt_id(q_list, nr, key, new_pid):
    base = str(nr)[:-2] if str(nr).endswith('-0') else str(nr)
    for q in q_list:
        if str(q.get('nr', '')) == base:
            q[key] = new_pid
            return True
        if update_q_prompt_id(q.get('questions', []), nr, key, new_pid):
            return True
    return False


def save_prompt_if_better(
    target_q, new_prompt, acc_baseline, acc_candidate,
    prompt_file, question_file_path, question_dictionary,
):
    if acc_baseline is None or acc_candidate is None:
        print('  [SKIP] Nav rezultātu (baseline vai candidate).')
        return 'no_data'

    if acc_candidate <= acc_baseline:
        print(f'  [SKIP] Candidate ({acc_candidate}%) nav labāks par baseline ({acc_baseline}%).')
        return 'not_better'

    delta = round(acc_candidate - acc_baseline, 2)
    print(f'  [SAVE] Candidate ir labāks par +{delta}% — saglabā uzvedni...')

    is_q0 = str(target_q).endswith('-0')
    prompt_key = 'prompt0-id' if is_q0 else 'prompt-id'

    q_data_target = find_question_data(question_dictionary, target_q)
    if q_data_target is None:
        print(f'  [ERROR] Jautājums {target_q} nav atrasts.')
        return 'error'

    current_prompt_id = q_data_target.get(prompt_key)
    if not current_prompt_id:
        print(f'  [ERROR] Jautājumam {target_q} nav "{prompt_key}" lauks.')
        return 'error'

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts_data = yaml.safe_load(f) or []

    is_default = any(p.get('id') == current_prompt_id and p.get('default') for p in prompts_data)

    with open(question_file_path, 'r', encoding='utf-8') as f:
        questions_data_fresh = yaml.load(f, Loader=yaml.BaseLoader) or []

    usage = count_usage_in_file(questions_data_fresh, current_prompt_id, prompt_key)
    print(
        f'  Uzvedne "{current_prompt_id}" lietojums: {usage} jautājumos, '
        f'noklusējums: {is_default}.'
    )

    if not is_default and usage <= 1:
        found = False
        for p in prompts_data:
            if p['id'] == current_prompt_id:
                p['prompt'] = new_prompt
                found = True
                break
        if not found:
            prompts_data.append({'id': current_prompt_id, 'prompt': new_prompt})

        with open(prompt_file, 'w', encoding='utf-8') as f:
            yaml.dump(prompts_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        print(f'  [OK] prompts.yaml "{current_prompt_id}" atjaunināta.')
        return 'updated'

    reason = 'noklusējums' if is_default else f'koplietota ({usage} jautājumi)'
    print(f'  Veido jaunu ID ({reason})...')

    safe_nr = str(target_q).replace('.', '_').replace('-', '_')
    new_id = f'p_for_{safe_nr}'
    existing_ids = {p['id'] for p in prompts_data}
    if new_id in existing_ids:
        v = 2
        while f'{new_id}_v{v}' in existing_ids:
            v += 1
        new_id = f'{new_id}_v{v}'

    prompts_data.append({'id': new_id, 'prompt': new_prompt})
    with open(prompt_file, 'w', encoding='utf-8') as f:
        yaml.dump(prompts_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f'  [OK] prompts.yaml — jauna uzvedne "{new_id}" pievienota.')

    if update_q_prompt_id(questions_data_fresh, target_q, prompt_key, new_id):
        with open(question_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                questions_data_fresh, f,
                allow_unicode=True, default_flow_style=False, sort_keys=False,
            )
        print(f'  [OK] questions.yaml jautājumam {target_q} "{prompt_key}" = "{new_id}".')
        return 'new_id'

    print(f'  [ERROR] Jautājums {target_q} nav atrasts questions.yaml!')
    return 'error'


async def optimize_one(
    target_q, config_dir, procurement_file_dir, answer_file_dir,
    report_dir, question_file_path, prompt_file,
    question_dictionary, supplementary_info, embedding, llm, extractor,
):
    print('\n' + '=' * 50)
    print(f'  OPTIMIZĒ JAUTĀJUMU: {target_q}')
    print('=' * 50)

    new_q = str(target_q).replace('.', '_')
    baseline_dir = report_dir / f'opt_{new_q}_baseline'
    baseline_csv = baseline_dir / 'report.csv'
    candidate_dir = report_dir / f'opt_{new_q}_candidate'
    candidate_csv = candidate_dir / 'report.csv'

    questions_to_process = questions_to_process_for(target_q)
    prompt_dictionary = get_prompt_dict(prompt_file, question_dictionary)
    ini_files = get_ini_files(config_dir, True, baseline_csv)

    # 1. SOLIS: Baseline
    await run_report(
        baseline_dir, baseline_csv, prompt_dictionary, ini_files,
        config_dir, procurement_file_dir, answer_file_dir,
        question_dictionary, supplementary_info, questions_to_process,
        embedding, llm, extractor, label=f'Baseline ({target_q})',
    )
    acc_b, n_b = compute_accuracy(baseline_csv, target_q)
    print(f'[Baseline] {target_q}: akuritāte = {acc_b}% ({n_b} iepirkumi)')

    # 2. SOLIS: Kļūdu analīze
    failures = extract_failures(baseline_csv, target_q)
    if failures.empty:
        print(f'  [SKIP] Jautājumam {target_q} nav kļūdu — uzvedne netiek mainīta.')
        return {
            'target_q': target_q, 'baseline_acc': acc_b, 'candidate_acc': None,
            'status': 'no_failures',
        }

    # 3. SOLIS: uzvednes labošana
    q_data = find_question_data(question_dictionary, target_q)
    q_text = q_data.get('question', q_data.get('question0', '')) if q_data else ''
    cur_prompt = prompt_dictionary.get(str(target_q), prompt_dictionary.get('0', ''))

    suggestion_request = build_suggestion_request(cur_prompt, q_text, failures)
    response = llm.complete(suggestion_request)

    match = re.search(r'<suggested_prompt>(.*?)</suggested_prompt>', response.text, re.DOTALL)
    if not match:
        print(f'  [SKIP] LLM neatgrieza uzvedni tagos priekš {target_q}.')
        return {
            'target_q': target_q, 'baseline_acc': acc_b, 'candidate_acc': None,
            'status': 'no_suggestion',
        }
    new_prompt = match.group(1).strip()

    # 4. SOLIS: Candidate
    candidate_prompt_dict = dict(prompt_dictionary)
    candidate_prompt_dict[str(target_q)] = new_prompt

    await run_report(
        candidate_dir, candidate_csv, candidate_prompt_dict, ini_files,
        config_dir, procurement_file_dir, answer_file_dir,
        question_dictionary, supplementary_info, questions_to_process,
        embedding, llm, extractor, label=f'Candidate ({target_q})',
    )
    acc_c, n_c = compute_accuracy(candidate_csv, target_q)
    print(f'[Candidate] {target_q}: akuritāte = {acc_c}% ({n_c} iepirkumi)')

    status = save_prompt_if_better(
        target_q, new_prompt, acc_b, acc_c,
        prompt_file, question_file_path, question_dictionary,
    )

    return {
        'target_q': target_q, 'baseline_acc': acc_b, 'candidate_acc': acc_c,
        'status': status,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--questions', nargs='+', help='Jautājumu numuri (piem., 39.4 40.5).')
    g.add_argument('--questions-file', help='Fails ar jautājumiem, viens katrā rindā.')
    p.add_argument('--config-dir', default='dev_config', help='config vai dev_config.')
    return p.parse_args()


def load_target_questions(args):
    if args.questions:
        return [q.strip() for q in args.questions if q.strip()]
    path = Path(args.questions_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


async def main():
    args = parse_args()
    target_questions = load_target_questions(args)

    print(f'Optimizēs {len(target_questions)} jautājumus: {target_questions}')

    question_file_path = PROJECT_ROOT / 'questions' / 'questions.yaml'
    prompt_file = PROJECT_ROOT / 'questions' / 'prompts.yaml'
    report_dir = PROJECT_ROOT / 'reports'
    config_dir = PROJECT_ROOT / args.config_dir
    procurement_file_dir = PROJECT_ROOT / 'cfla_files'
    answer_file_dir = PROJECT_ROOT / 'answers'

    embedding = HuggingFaceEmbedding(
        model_name=EMBEDDING_CONF['embeddingmodel'], trust_remote_code=True,
    )
    llm = AzureOpenAI(
        azure_deployment='gpt-4o',
        azure_endpoint=os.environ.get('AZURE_ENDPOINT', ''),
        temperature=0.0,
        api_version=os.environ.get('AZURE_OPENAI_VERSION', ''),
        api_key=os.environ.get('AZURE_OPENAI_KEY', ''),
        timeout=120, max_retries=3, top_p=0.0001,
    )
    extractor = Extractor()
    question_dictionary = get_questions(question_file_path)
    supplementary_info = get_supplementary_info()

    results = []
    for tq in target_questions:
        try:
            res = await optimize_one(
                tq, config_dir, procurement_file_dir, answer_file_dir,
                report_dir, question_file_path, prompt_file,
                question_dictionary, supplementary_info, embedding, llm, extractor,
            )
        except Exception as e:
            print(f'[ERROR] Jautājums {tq}: {type(e).__name__}: {e}')
            res = {'target_q': tq, 'baseline_acc': None, 'candidate_acc': None, 'status': f'exception: {e}'}
        results.append(res)

    print('\n' + '=' * 70)
    print('KOPSAVILKUMS')
    print('=' * 70)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    summary_path = report_dir / 'optimize_loop_summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    print(f'\nKopsavilkums saglabāts: {summary_path}')


if __name__ == '__main__':
    asyncio.run(main())
