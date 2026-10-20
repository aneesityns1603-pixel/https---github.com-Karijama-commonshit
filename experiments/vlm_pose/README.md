# Проверка предобученных моделей на pose estimation

Обучение и дообучение не предусмотрены. Это подготовка эксперимента и оценщик,
а не уже выполненный эксперимент VLM или запуск MV-SAM3D.

## Протокол

1. Разместить исходный CAD по GT; спроецировать **все вершины** на все валидные RGB-кадры.
2. Выбрать 2–3 кадра по числу проекций внутри изображения с положительным Z.
3. Дополнительно можно ранжировать по приблизительной видимости и ограничить близость ракурсов.
4. Передать внешней модели исходные RGB, представление CAD и параметры камер.
5. Сохранить первичный прогноз VLM. Отдельно можно сохранить результат геометрической подгонки.
6. Сравнить оба результата с GT. Не выдавать точность после подгонки за точность самой VLM.

Первый этап использует **известные GT-камеры**. Это явное упрощение для изоляции
ошибки объектной позы; восстановление неизвестных камер является отдельным экспериментом.
GT объекта, его bbox, ключевые точки, облако скана и диагностические наложения
не входят в model_input. Изображения отобраны по GT, поэтому это oracle-assisted benchmark.
Нельзя предоставлять модели доступ ко всему репозиторию с GT вместо ограниченного пакета.

`run_qwen.py` создаёт четыре изображения CAD с подписанными локальными осями
в одном листе, добавляет наблюдения и камеры в запрос локального Ollama.
Идентификация целевого экземпляра среди одинаковых предметов остаётся неоднозначной:
числовой instance_id сам по себе не обозначает предмет для VLM. Промпт требует
вернуть ambiguous, если выбрать экземпляр нельзя.
Разметку целевого bbox/маски из GT можно вводить только как отдельный явно обозначенный oracle-режим.

## Запуск

Проверено в имеющемся Python 3.10; нужны numpy, scipy, Pillow (уже доступны).
Для отдельной среды: `python -m pip install numpy scipy Pillow`.
Команды выполняются из корня scannet5:

```powershell
python -m unittest discover -s experiments/vlm_pose -p 'test_*.py' -v
python experiments/vlm_pose/select_views.py --scenes all
python experiments/vlm_pose/select_views.py --scenes all --score approx_unoccluded_vertices --min-angle 15 --output experiments/vlm_pose/results_diverse
```

Первый запуск буквально выбирает top-3 по количеству вершин, без ограничения угла.
Второй — дополнительный вариант, а не подмена исходного правила.
Если трёх подходящих ракурсов нет, экспортируется меньше; дубликатами набор не заполняется.

Результаты: HTML-галерея для каждой сцены, таблица всех кадров в selection.json,
RGB и камеры в model_input, GT в отдельном evaluation_only.json.
Красные проекции в галерее — только диагностика; они не являются масками сегментации.

## Ограничения видимости

Попадание вершины в кадр не означает видимость. Дополнительная метрика использует
z-buffer вершин самого CAD и z-buffer **готового облака скана** в RGB-системе.
Размер ячейки по умолчанию 4 пикселя; minimum-filter радиусом 1 заполняет небольшие
пробелы облака; допуск по глубине 0.10 м учитывает неточное совмещение CAD.
Это приближённая эвристика: нет треугольного rasterization/ray tracing,
разреженность даёт неизвестную видимость, фильтр может закрывать реальные зазоры.
Неизвестная глубина допускается в score, но её количество явно записывается отдельно.
Depth PNG не используются: точной K_depth в наборе нет.
Значения числа вершин сравниваются только между кадрами **одного** экземпляра:
плотность вершин у разных CAD сильно отличается.

## Оценка внешнего прогноза

Формат prediction.json:

```json
{"translation":[0,0,0],"rotation_wxyz":[1,0,0,0],"scale_xyz":[1,1,1]}
```

Это только пример формата, не результат модели.
Соглашение: `p_world = R(q_wxyz) @ diag(scale_xyz) @ p_cad + translation`, метры ScanNet.

```powershell
python experiments/vlm_pose/evaluate_pose.py --prediction prediction.json --ground-truth experiments/vlm_pose/results/scene0084_01/instance_00/evaluation_only.json --output metrics.json
```

Метрики: перенос, сырой угол SO(3), относительные ошибки масштаба,
средняя ошибка соответствующих вершин и односторонняя ближайшая вершинная дистанция.
Последняя не равна точному поверхностному Chamfer. Сырой угол не учитывает симметрии;
его нельзя использовать как единственную метрику для симметричных объектов.
Оценщик не изменяет прогноз и не запускает оптимизацию.

## MV-SAM3D: что можно использовать

Изучены исходники:
- https://github.com/devinli123/MV-SAM3D
- https://github.com/devinli123/MV-SAM3D/blob/main/run_inference_weighted.py
- https://github.com/devinli123/MV-SAM3D/blob/main/sam3d_objects/pose_align/pose_optimization.py
- https://github.com/devinli123/MV-SAM3D/blob/main/preprocessing/build_mvsam3d_dataset.py

Реконструкция использует предобученные SAM 3D Objects и DA3, preprocessing — SAM3.
Входные маски — RGBA PNG с foreground в alpha.
PoseOptimizer подгоняет сгенерированный canonical mesh к выделенному по маскам
облаку DA3: односторонний target-to-source Chamfer, регуляризация, Adam.
Оптимизируются quaternion и translation; единый scale опционален и по умолчанию фиксирован.
Этот scale не покрывает независимые sx/sy/sz нашего GT.
Преобразования SAM3D/PyTorch3D/DA3 требуют адаптации к соглашениям ScanNet.
При генерации геометрии нужно отдельно согласовать canonical frame с исходным CAD:
ошибку позы сгенерированного меша нельзя напрямую сравнивать с CAD GT.

Разумное сравнение: VLM direct pose; MV-SAM3D без/с pose optimization;
VLM initialization + одинаковая геометрическая подгонка. Реконструкцию и позу
известного CAD оценивать отдельно. MV-SAM3D не является frontier VLM baseline.

## Qwen 3.5 9B на другом компьютере

Выбрана пользователем Qwen 3.5 9B, локальный инференс. Установите Ollama
на компьютере с достаточной RAM/VRAM и загрузите модель:

```powershell
ollama pull qwen3.5:9b
ollama list
```

Документация: https://ollama.com/library/qwen3.5:9b и https://docs.ollama.com/api/chat.
Зависимости адаптера: numpy, scipy, Pillow, requests. Обучение не требуется.
Данные scannet5 нужно перенести на тот же компьютер отдельно от Git-репозитория.
Пример из корня репозитория (замените D:/datasets/scannet5 своим путём):

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s experiments/vlm_pose -p 'test_*.py' -v
python experiments/vlm_pose/select_views.py --root D:/datasets/scannet5 --scenes scene0084_01 --score approx_unoccluded_vertices --min-angle 15 --output results
python experiments/vlm_pose/run_qwen.py --root D:/datasets/scannet5 --input results/scene0084_01/instance_02/model_input/input.json --output runs/qwen_instance02
python experiments/vlm_pose/evaluate_pose.py --root D:/datasets/scannet5 --prediction runs/qwen_instance02/prediction.json --ground-truth results/scene0084_01/instance_02/evaluation_only.json --output runs/qwen_instance02/metrics.json
```

Ollama должен быть запущен локально на 127.0.0.1:11434; адаптер не включает сервер
и не скачивает модель автоматически. Поддерживается только loopback — запускайте
скрипт на компьютере с моделью. Для буквального top-3 уберите --score и --min-angle.
Для всех объектов: --scenes all. Начать стоит с одного объекта, затем повторять
run_qwen для каждого model_input, фиксируя отказы и неоднозначные ответы.
Случаи с менее чем двумя выбранными ракурсами адаптер отклоняет.
Оценщик запускайте только для status=ok. Отказы нельзя исключать из отчёта о coverage.

Сохраняются request.json, полный response.json, метаданные установленной модели
и проверенный prediction.json. Новый output должен отсутствовать: это предотвращает
смешение результатов разных запусков. Параметры: temperature=0, seed=42,
context=16384, max_side=896, think=false. При уменьшении изображений пересчитывается K.
Можно сначала добавить --prepare-only, чтобы проверить запрос без инференса
(для реального прогона затем укажите другой output).
Локально выполнен только prepare-only, включая создание и визуальную проверку CAD-превью.
Это не результат VLM. Прогнозов и численных оценок качества модели пока нет.

## Что пока не запущено

MV-SAM3D и его веса не установлены, инференс не выполнялся.
На текущем компьютере 5.9 ГБ RAM и MX250 с 2 ГБ VRAM; инференс переносится
на другой компьютер по указанию пользователя.
Тесты проверяют геометрию и оценщик на известных числах, а не качество моделей.
