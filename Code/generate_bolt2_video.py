import cv2
out = cv2.VideoWriter('bhaag_bolt_bhaag.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), 30, (480, 270))

#/home/kulbir/Desktop/Lucas_Kanade_Tracker/Code
#out = cv2.VideoWriter('output_yellow.mp4', cv2.VideoWriter_fourcc(*'XVID'), 5, (640, 480))

imgArray = []
for ind in range(0,293,1):
    print(ind)

    plt_name = './output/Bolt2/' + str(ind) + '.jpg'
    plot_img = cv2.imread(plt_name)
    imgArray.append(plot_img)

    # read the image stored using cv2
    # plot_img.shape = gives dimension of the frame
    # print('frame', plot_img.shape)

    # write the image in the video

for i in range(len(imgArray)):
    out.write(imgArray[i])

out.release()
