# Schema Gaps

Live tools whose registered argument schema cannot express an intent the
training corpus contains. The corpus argument is preserved verbatim so
the model keeps the intent; the handler should be extended to accept it.

| Tool | Missing capability | Corpus calls |
| :--- | :--- | ---: |
| `browser_switch_tab` | corpus targets tabs by name ('switch to docs tab'); schema only accepts tab_index:int | 112 |

