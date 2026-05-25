# Izguves papildinātas ģenerēšanas metodes dokumentu atbilstības pārbaudei

Šis repozitorijs, papildus VPP-CFLA projektā izveidotajam, satur bakalaura darba **"Izguves papildinātas ģenerēšanas metodes dokumentu atbilstības pārbaudei"** praktisko daļu.

Bakalaura darbā ir izstrādāts automatizēts uzvedņu (*angliski - prompts*) optimizācijas risinājums, kas iteratīvi uzlabo lielo valodas modeļu (LLM) atbilžu akurātumu Centrālās finanšu un līgumu aģentūras (CFLA) iepirkumu pārbaudes jautājumiem nolieguma formā. Risinājums izmanto pašu LLM kā uzvedņu inženieri, kas analizē kļūdainās atbildes un ģenerē uzlabotas kandidātuzvednes.

## VPP-CFLA projekts

Šis repozitorijs ir sazarojums (*angliski - fork*) no [VPP-CFLA projekta prototipa](https://github.com/raivisskadins/vpp-cfla), kuru izstrādājusi VPP-CFLA projekta komanda, kurā piedalījusies arī šī bakalaura darba autore. Tas ir papildināts ar uzvedņu automatizētas optimizācijas funkcionalitāti, kas izstrādāta bakalaura darba ietvaros.

## Galvenie risinājuma faili

- **`scripts/optimize_prompts_loop.py`** — automatizēts Python skripts uzvedņu iteratīvai optimizācijai. Atbilst bakalaura darba 5.2. nodaļā aprakstītajam algoritmam. Izsaukšanas piemērs:
```bash
  python -m scripts.optimize_prompts_loop --questions 39 39.20
```
- **`PromptOptimizer.ipynb`** — interaktīvs prototips Jupyter Notebook formātā uzvedņu optimizācijai. Atbilst bakalaura darba 5.2. nodaļā aprakstītajam interaktīvajam prototipam.
- **`ProjectProcurementReview.ipynb`** - akurātuma novērtēšanas un atskaišu ģenerēšanas piezīmjgrāmata izstrādes un validācijas kopām.
- **`questions/`** — mape ar pārbaudes lapu jautājumu un uzvednes failiem YAML formātā.

## Papildu informācija

Detalizēta uzstādīšanas un lietošanas instrukcija pieejama failā [USAGE.md](USAGE.md).
