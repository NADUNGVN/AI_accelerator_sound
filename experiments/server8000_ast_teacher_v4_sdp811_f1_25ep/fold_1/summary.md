# AST Teacher Fine-Tune: fold_1

## Result

| Metric | Value |
|---|---:|
| Completed epochs | 15 |
| Early stopped | True |
| Best epoch | 5 |
| Best val | 89.49% |
| Best test | 88.51% |
| Final val | 89.49% |
| Final test | 90.46% |

## Best-Test Per Class

| Class | Support | Correct | Accuracy | Main wrong |
|---|---:|---:|---:|---|
| air_conditioner | 100 | 90 | 90.00% | drilling:5, engine_idling:4, gun_shot:1 |
| car_horn | 40 | 39 | 97.50% | air_conditioner:1 |
| children_playing | 100 | 90 | 90.00% | street_music:5, siren:4, dog_bark:1 |
| dog_bark | 100 | 93 | 93.00% | drilling:3, air_conditioner:1, children_playing:1, gun_shot:1 |
| drilling | 100 | 92 | 92.00% | jackhammer:7, car_horn:1 |
| engine_idling | 100 | 68 | 68.00% | jackhammer:31, air_conditioner:1 |
| gun_shot | 37 | 37 | 100.00% |  |
| jackhammer | 98 | 73 | 74.49% | air_conditioner:23, drilling:2 |
| siren | 93 | 93 | 100.00% |  |
| street_music | 102 | 95 | 93.14% | children_playing:4, air_conditioner:2, car_horn:1 |

## Weak Source Groups

| Class | fsID | Correct/support | Acc | Main wrong |
|---|---:|---:|---:|---|
| engine_idling | 144007 | 3/34 | 8.82% | jackhammer:31 |
| air_conditioner | 80589 | 1/5 | 20.00% | drilling:4 |
| street_music | 98681 | 2/6 | 33.33% | children_playing:3, air_conditioner:1 |
| jackhammer | 177537 | 22/45 | 48.89% | air_conditioner:23 |
| children_playing | 76266 | 4/8 | 50.00% | street_music:3, dog_bark:1 |
| street_music | 136558 | 3/6 | 50.00% | children_playing:1, car_horn:1, air_conditioner:1 |
| drilling | 156362 | 9/16 | 56.25% | jackhammer:6, car_horn:1 |
| children_playing | 160016 | 5/7 | 71.43% | siren:2 |
| children_playing | 151149 | 6/8 | 75.00% | street_music:2 |
| children_playing | 155219 | 6/8 | 75.00% | siren:2 |

## History

| Epoch | Train | Val | Test | Train loss | Seconds |
|---:|---:|---:|---:|---:|---:|
| 1 | 78.36% | 86.03% | 86.78% | 0.9799 | 44.3 |
| 2 | 91.35% | 87.53% | 87.70% | 0.5241 | 44.6 |
| 3 | 93.58% | 85.57% | 88.51% | 0.4590 | 45.1 |
| 4 | 96.53% | 89.95% | 87.70% | 0.3793 | 155.3 |
| 5 | 98.90% | 89.49% | 88.51% | 0.3110 | 155.8 |
| 6 | 99.43% | 89.03% | 89.08% | 0.2919 | 155.6 |
| 7 | 99.54% | 89.03% | 87.13% | 0.2827 | 155.7 |
| 8 | 99.70% | 88.57% | 89.66% | 0.2760 | 155.7 |
| 9 | 99.70% | 88.91% | 90.46% | 0.2745 | 155.7 |
| 10 | 99.74% | 89.38% | 90.69% | 0.2711 | 155.7 |
| 11 | 99.69% | 89.26% | 89.54% | 0.2699 | 155.5 |
| 12 | 99.74% | 89.03% | 90.57% | 0.2707 | 155.4 |
| 13 | 99.76% | 89.38% | 90.46% | 0.2679 | 155.3 |
| 14 | 99.73% | 88.80% | 89.89% | 0.2697 | 155.4 |
| 15 | 99.83% | 89.49% | 90.46% | 0.2676 | 155.3 |