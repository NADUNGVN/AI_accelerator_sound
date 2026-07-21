# Result report template (one experiment)

Copy into `docs/experiments/notes/` or lab log.

```markdown
## Exp: <exp_name>
- Date:
- Branch (code): research/fpga-1dcnn-90acc @ <sha>
- Results branch:
- Config:
- Protocol / seed:
- Folds / epochs:
- Model / params:
- Teacher/KD: yes/no

### Data integrity
- train/val/test counts:
- source overlap train–test:
- uses_validation:

### Metrics
| Metric | Value |
|---|---:|
| best val | |
| **best-val test** | |
| last snapshot | |
| ensemble | |
| worst class | |

### vs H0 (76.90% best-val test server fold1)
- Δ best-val test: 
- Decision: IMPROVED / PARTIAL / REJECTED

### Errors
- Top confusions:
- Did we fix the intended failure mode?

### Next
- Re-run? no / yes with hypothesis:
- Registry status:
```
