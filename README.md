# Money-dector-for-the-blind
Moner dector project for the blind





running things:

for making config file:
# If you have a single images/ and labels/ directory, the script will automatically split them
python ultralytics_money_detector.py create-dataset --dataset_dir dataset --output dataset.yaml

to split datasets:
python simple_split.py dataset

to debug split:
python debug_dataset.py dataset

to run training:
python ultralytics_money_detector.py train --data dataset.yaml --epochs 100
