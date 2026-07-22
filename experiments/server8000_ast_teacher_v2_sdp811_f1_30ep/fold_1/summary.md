# AST Teacher Fine-Tune: fold_1

## Result

| Metric | Value |
|---|---:|
| Completed epochs | 15 |
| Early stopped | True |
| Best epoch | 3 |
| Best val | 91.80% |
| Best test | 89.43% |
| Final val | 88.68% |
| Final test | 89.89% |

## Best-Test Per Class

| Class | Support | Correct | Accuracy | Main wrong |
|---|---:|---:|---:|---|
| air_conditioner | 100 | 96 | 96.00% | engine_idling:4 |
| car_horn | 40 | 40 | 100.00% |  |
| children_playing | 100 | 92 | 92.00% | street_music:5, siren:2, dog_bark:1 |
| dog_bark | 100 | 89 | 89.00% | children_playing:6, drilling:3, air_conditioner:2 |
| drilling | 100 | 88 | 88.00% | car_horn:9, jackhammer:3 |
| engine_idling | 100 | 61 | 61.00% | jackhammer:34, drilling:3, air_conditioner:2 |
| gun_shot | 37 | 37 | 100.00% |  |
| jackhammer | 98 | 87 | 88.78% | air_conditioner:8, drilling:3 |
| siren | 93 | 91 | 97.85% | air_conditioner:1, gun_shot:1 |
| street_music | 102 | 97 | 95.10% | children_playing:3, air_conditioner:1, car_horn:1 |

## Weak Source Groups

| Class | fsID | Correct/support | Acc | Main wrong |
|---|---:|---:|---:|---|
| engine_idling | 144007 | 0/34 | 0.00% | jackhammer:34 |
| drilling | 156362 | 5/16 | 31.25% | car_horn:9, jackhammer:2 |
| street_music | 98681 | 3/6 | 50.00% | children_playing:2, air_conditioner:1 |
| children_playing | 76266 | 5/8 | 62.50% | street_music:2, dog_bark:1 |
| street_music | 136558 | 4/6 | 66.67% | children_playing:1, car_horn:1 |
| children_playing | 160016 | 5/7 | 71.43% | street_music:2 |
| children_playing | 155219 | 6/8 | 75.00% | siren:2 |
| jackhammer | 177537 | 37/45 | 82.22% | air_conditioner:8 |
| jackhammer | 188824 | 11/13 | 84.62% | drilling:2 |
| children_playing | 151149 | 7/8 | 87.50% | street_music:1 |

## History

| Epoch | Train | Val | Test | Train loss | Seconds |
|---:|---:|---:|---:|---:|---:|
| 1 | 83.03% | 86.84% | 87.01% | 0.7763 | 44.2 |
| 2 | 92.58% | 86.49% | 88.05% | 0.4318 | 44.5 |
| 3 | 96.34% | 91.80% | 89.43% | 0.3284 | 154.7 |
| 4 | 99.10% | 90.18% | 90.69% | 0.2484 | 155.4 |
| 5 | 99.44% | 88.80% | 91.38% | 0.2284 | 155.6 |
| 6 | 99.59% | 89.26% | 91.49% | 0.2228 | 155.4 |
| 7 | 99.59% | 88.34% | 89.66% | 0.2225 | 155.0 |
| 8 | 99.71% | 89.49% | 90.57% | 0.2194 | 155.0 |
| 9 | 99.71% | 89.95% | 90.80% | 0.2179 | 154.8 |
| 10 | 99.76% | 87.30% | 90.23% | 0.2157 | 155.0 |
| 11 | 99.73% | 89.15% | 89.77% | 0.2144 | 154.8 |
| 12 | 99.74% | 89.61% | 90.57% | 0.2145 | 154.8 |
| 13 | 99.74% | 89.61% | 87.93% | 0.2117 | 154.9 |
| 14 | 99.69% | 87.41% | 91.95% | 0.2154 | 155.1 |
| 15 | 99.79% | 88.68% | 89.89% | 0.2125 | 154.9 |