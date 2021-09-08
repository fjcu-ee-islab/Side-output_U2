from math import log10, sqrt
import cv2
import numpy as np
import os

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        return 100

    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
def main():
    gt_file = '/home/ubuntu/Downloads/Side-output_U2-main/data/adobe/adobe_test/test_1'
    output = '/home/ubuntu/Downloads/Side-output_U2-main/result_folder/test_adobe_result_1'
    av_psnr = 0
    gt_path = os.listdir(gt_file)
    count=0
    for i in gt_path:
        #print()
        #print(i)

        gt = []
        out =[]

        image_path = os.path.join(gt_file, i)
        output_path = os.path.join(output, i)

        gt_image = os.listdir(image_path)
        gt_image.sort()

        for k in gt_image:
            gt.append(k)

        pred = os.listdir(output_path)
        pred.sort()

        for k in pred:
            out.append(k)
        #print(gt)
        #print(out)        

        #print(len(pred))
        for j in range(1,len(pred)-1):
            #print(j)
            #print(image_path)
            #print(output_path)
            #print(gt[j])
            #print(out[j])

            original = cv2.imread(os.path.join(image_path, gt[j]))
            compressed = cv2.imread(os.path.join(output_path, out[j]), 1)
            value = PSNR(original, compressed)
            #print(value)
            count=count+1
            av_psnr = av_psnr+value


        gt = []
        out =[]
    #print(len(gt_path))
    #print(len(pred))
    #print(count)
    av_psnr = av_psnr / count
    print(f"60fps PSNR value is {av_psnr} dB")

    gt_file = '/home/ubuntu/Downloads/Side-output_U2-main/data/adobe/adobe_test/test_3'
    output = '/home/ubuntu/Downloads/Side-output_U2-main/result_folder/test_adobe_result_3'
    av_psnr = 0
    gt_path = os.listdir(gt_file)
    count=0
    for i in gt_path:
        #print()
        #print(i)

        gt = []
        out =[]

        image_path = os.path.join(gt_file, i)
        output_path = os.path.join(output, i)

        gt_image = os.listdir(image_path)
        gt_image.sort()

        for k in gt_image:
            gt.append(k)

        pred = os.listdir(output_path)
        pred.sort()

        for k in pred:
            out.append(k)
        #print(gt)
        #print(out)        

        #print(len(pred))
        for j in range(1,len(pred)-1):
            #print(j)
            #print(image_path)
            #print(output_path)
            #print(gt[j])
            #print(out[j])

            original = cv2.imread(os.path.join(image_path, gt[j]), 1)
            compressed = cv2.imread(os.path.join(output_path, out[j]), 1)
            value = PSNR(original, compressed)
            #print(value)
            count=count+1
            av_psnr = av_psnr+value


        gt = []
        out =[]
    #print(len(gt_path))
    #print(len(pred))
    #print(count)
    av_psnr = av_psnr / count
    print(f"120fps PSNR value is {av_psnr} dB")

    gt_file = '/home/ubuntu/Downloads/Side-output_U2-main/data/adobe/adobe_test/test_7'
    output = '/home/ubuntu/Downloads/Side-output_U2-main/result_folder/test_adobe_result_7'
    av_psnr = 0
    gt_path = os.listdir(gt_file)
    count=0
    for i in gt_path:
        #print()
        #print(i)

        gt = []
        out =[]

        image_path = os.path.join(gt_file, i)
        output_path = os.path.join(output, i)

        gt_image = os.listdir(image_path)
        gt_image.sort()

        for k in gt_image:
            gt.append(k)

        pred = os.listdir(output_path)
        pred.sort()

        for k in pred:
            out.append(k)
        #print(gt)
        #print(out)        

        #print(len(pred))
        for j in range(1,len(pred)-1):
            #print(j)
            #print(image_path)
            #print(output_path)
            #print(gt[j])
            #print(out[j])




            original = cv2.imread(os.path.join(image_path, gt[j]), 1)
            compressed = cv2.imread(os.path.join(output_path, out[j]), 1)
            value = PSNR(original, compressed)
            #print(value)
            count=count+1
            av_psnr = av_psnr+value


        gt = []
        out =[]
    #print(len(gt_path))
    #print(len(pred))
    #print(count)
    av_psnr = av_psnr / count
    print(f"240fps PSNR value is {av_psnr} dB")


if __name__ == "__main__":
    main()
