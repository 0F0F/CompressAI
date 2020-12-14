import os

for quality in [1,2,3]:
    for img in range(1, 25):
        print("python model.py encode /data/dataset/test/kodim{img:02d}.png -m mse -q {q} 2 "
                  "-o /data/result/{q}_{img:02d}.out".format(img=img,q=quality))
        print("python model.py decode /data/result/{q}_{img:02d}.out -o /data/result/{q}_{img:02d}.jpg".format(img=img,q=quality))