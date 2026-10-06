# Task 3 data (shared, not committed)

Download the class Kaggle competition data (link in the lab spec) and unzip so that:

```
task3_gan/data/monet_jpg/*.jpg   # ~300 Monet paintings, 256x256  (domain A)
task3_gan/data/photo_jpg/*.jpg   # ~7,000 photos, 256x256         (domain B)
```

With the Kaggle CLI (`pip install kaggle`, API token in `~/.kaggle/kaggle.json`, competition rules accepted on the website):

```bash
kaggle competitions download -c <competition-slug> -p task3_gan/data
unzip -q task3_gan/data/<competition-slug>.zip "monet_jpg/*" "photo_jpg/*" -d task3_gan/data
```
