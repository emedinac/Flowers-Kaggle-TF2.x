tfds new dataset # Init with this line
python3 modify_database.py --path_data="Database/flowers/" --path_mod="db_gen_modification.txt"
# Run DB generator
python3 change_default_dataset_classes.py --validation=0.5 --path_data="Database/flowers/" --output_path="Database/flowers/"
# Run an automatic code here to modify dataset
cd dataset
tfds build --overwrite --data_dir="../dataset/"
cd ..
