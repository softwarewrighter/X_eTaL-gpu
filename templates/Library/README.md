# __NAME__

__SUMMARY__

```
"__ALIAS__" u_se< "__NAME__"
```

| Directory | What |
| --------- | ---- |
| [`src/`](src/) | the library, `__NAME__.xtl` |
| [`docs/`](docs/README.md) | the reference: every function, its type, examples, provenance |
| [`demos/`](demos/) | programs that use it |
| [`tests/`](tests/) | reg-rs baselines: the test programs, the pinned export types (`types.rgt`) and the demos (`demo-*.rgt`) |

```bash
just demo __NAME__         # run its demos
just run __NAME__          # run its test programs
just test-lib __NAME__     # check every baseline
```
