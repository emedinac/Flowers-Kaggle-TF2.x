rm -rf flowers
tfds new flowers # Init with this line
python3 modify_database.py --dataset_name="flowers" --path_data="Database/flowers/" --path_mod="db_gen_modification.txt"
# Run DB generator
python3 change_default_dataset_classes.py --validation=0.5 --path_data="Database/flowers/" --output_path="Database/flowers/" --seed=100
# Run an automatic code here to modify dataset
cd flowers
tfds build --overwrite #--data_dir="../flowers/" # check details for --data_dir
cd ..
