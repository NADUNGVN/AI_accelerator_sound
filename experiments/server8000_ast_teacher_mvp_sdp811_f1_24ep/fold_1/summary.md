# AST Teacher Fine-Tune: fold_1

## Result

| Metric | Value |
|---|---:|
| Completed epochs | 9 |
| Early stopped | True |
| Best epoch | 3 |
| Best val | 88.45% |
| Best test | 88.28% |
| Final val | 84.76% |
| Final test | 88.39% |

## Best-Test Per Class

| Class | Support | Correct | Accuracy | Main wrong |
|---|---:|---:|---:|---|
| air_conditioner | 100 | 78 | 78.00% | drilling:13, engine_idling:7, car_horn:1, gun_shot:1 |
| car_horn | 40 | 39 | 97.50% | engine_idling:1 |
| children_playing | 100 | 95 | 95.00% | siren:2, street_music:2, dog_bark:1 |
| dog_bark | 100 | 93 | 93.00% | children_playing:3, air_conditioner:1, engine_idling:1, siren:1 |
| drilling | 100 | 84 | 84.00% | jackhammer:12, car_horn:2, children_playing:1, engine_idling:1 |
| engine_idling | 100 | 66 | 66.00% | jackhammer:34 |
| gun_shot | 37 | 37 | 100.00% |  |
| jackhammer | 98 | 90 | 91.84% | air_conditioner:7, children_playing:1 |
| siren | 93 | 92 | 98.92% | gun_shot:1 |
| street_music | 102 | 94 | 92.16% | children_playing:5, drilling:2, car_horn:1 |

## Weak Source Groups

| Class | fsID | Correct/support | Acc | Main wrong |
|---|---:|---:|---:|---|
| engine_idling | 144007 | 0/34 | 0.00% | jackhammer:34 |
| drilling | 156362 | 2/16 | 12.50% | jackhammer:10, car_horn:2, children_playing:1, engine_idling:1 |
| street_music | 98681 | 1/6 | 16.67% | children_playing:4, drilling:1 |
| air_conditioner | 80589 | 1/5 | 20.00% | drilling:4 |
| air_conditioner | 146690 | 13/25 | 52.00% | drilling:9, engine_idling:2, car_horn:1 |
| children_playing | 76266 | 5/8 | 62.50% | street_music:2, dog_bark:1 |
| street_music | 136558 | 4/6 | 66.67% | children_playing:1, car_horn:1 |
| children_playing | 155219 | 6/8 | 75.00% | siren:2 |
| street_music | 109263 | 4/5 | 80.00% | drilling:1 |
| jackhammer | 177537 | 37/45 | 82.22% | air_conditioner:7, children_playing:1 |

## History

| Epoch | Train | Val | Test | Train loss | Seconds |
|---:|---:|---:|---:|---:|---:|
| 1 | 81.75% | 87.30% | 85.17% | 0.7674 | 47.4 |
| 2 | 95.67% | 85.80% | 81.72% | 0.2998 | 160.7 |
| 3 | 98.70% | 88.45% | 88.28% | 0.2195 | 161.9 |
| 4 | 99.19% | 86.14% | 89.20% | 0.2086 | 161.8 |
| 5 | 99.37% | 88.11% | 88.51% | 0.2015 | 161.9 |
| 6 | 99.37% | 84.76% | 88.62% | 0.2031 | 161.7 |
| 7 | 99.61% | 86.26% | 89.54% | 0.1952 | 162.3 |
| 8 | 99.37% | 86.61% | 89.54% | 0.1988 | 184.7 |
| 9 | 99.39% | 84.76% | 88.39% | 0.1994 | 170.2 |