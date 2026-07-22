# AST Teacher Fine-Tune: fold_1

## Result

| Metric | Value |
|---|---:|
| Completed epochs | 9 |
| Early stopped | True |
| Best epoch | 1 |
| Best val | 89.84% |
| Best test | 89.77% |
| Final val | 88.68% |
| Final test | 91.26% |

## Best-Test Per Class

| Class | Support | Correct | Accuracy | Main wrong |
|---|---:|---:|---:|---|
| air_conditioner | 100 | 87 | 87.00% | engine_idling:10, drilling:3 |
| car_horn | 40 | 39 | 97.50% | engine_idling:1 |
| children_playing | 100 | 94 | 94.00% | street_music:3, siren:2, dog_bark:1 |
| dog_bark | 100 | 95 | 95.00% | siren:2, air_conditioner:1, children_playing:1, engine_idling:1 |
| drilling | 100 | 89 | 89.00% | jackhammer:9, car_horn:2 |
| engine_idling | 100 | 65 | 65.00% | jackhammer:33, air_conditioner:2 |
| gun_shot | 37 | 37 | 100.00% |  |
| jackhammer | 98 | 87 | 88.78% | air_conditioner:10, drilling:1 |
| siren | 93 | 92 | 98.92% | air_conditioner:1 |
| street_music | 102 | 96 | 94.12% | children_playing:5, car_horn:1 |

## Weak Source Groups

| Class | fsID | Correct/support | Acc | Main wrong |
|---|---:|---:|---:|---|
| engine_idling | 144007 | 1/34 | 2.94% | jackhammer:33 |
| street_music | 98681 | 2/6 | 33.33% | children_playing:4 |
| drilling | 156362 | 6/16 | 37.50% | jackhammer:8, car_horn:2 |
| children_playing | 76266 | 5/8 | 62.50% | street_music:2, dog_bark:1 |
| street_music | 136558 | 4/6 | 66.67% | children_playing:1, car_horn:1 |
| children_playing | 155219 | 6/8 | 75.00% | siren:2 |
| air_conditioner | 146690 | 19/25 | 76.00% | engine_idling:6 |
| jackhammer | 177537 | 35/45 | 77.78% | air_conditioner:10 |
| car_horn | 185436 | 7/8 | 87.50% | engine_idling:1 |
| children_playing | 151149 | 7/8 | 87.50% | street_music:1 |

## History

| Epoch | Train | Val | Test | Train loss | Seconds |
|---:|---:|---:|---:|---:|---:|
| 1 | 99.26% | 89.84% | 89.77% | 0.1734 | 152.4 |
| 2 | 99.49% | 89.49% | 90.00% | 0.1639 | 154.8 |
| 3 | 99.71% | 88.11% | 91.15% | 0.1551 | 155.3 |
| 4 | 99.71% | 89.26% | 90.11% | 0.1564 | 155.4 |
| 5 | 99.73% | 89.38% | 90.57% | 0.1536 | 155.4 |
| 6 | 99.63% | 89.38% | 90.34% | 0.1539 | 155.4 |
| 7 | 99.73% | 89.49% | 90.80% | 0.1518 | 155.3 |
| 8 | 99.80% | 89.49% | 91.38% | 0.1495 | 155.4 |
| 9 | 99.81% | 88.68% | 91.26% | 0.1489 | 155.4 |