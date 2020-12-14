import os

for quality in [1,2,3]:
    for img in range(1, 25):
        os.system("python /data/compressai/examples/model.py encode /data/dataset/eval/kodim{img:02d}.png -m mse -q {q}  "
                  "-o /data/result/kodim{img:02d}_{q}.out".format(img=img,q=quality))
        os.system("python /data/compressai/examples/model.py decode /data/result/kodim{img:02d}_{q}.out -o /data/result/kodim{img:02d}_{q}.jpg".format(img=img,q=quality))
        os.system("python /data/compressai/examples/model.py decode /data/result/kodim{img:02d}_{q}.out -o /data/result/kodim{img:02d}_{q}.png".format(img=img,q=quality))
