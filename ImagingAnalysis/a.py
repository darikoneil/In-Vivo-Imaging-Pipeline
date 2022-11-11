# Video = PreProcessing.loadRawBinary("", "", "D:\\EM0122\\Encoding\\Imaging\\30Hz\\denoised")[0:1000]
# Stat = np.load("D:\\EM0122\\Encoding\\Imaging\\30Hz\\suite2p\\plane0\\stat.npy", allow_pickle=True)
# IsCell = np.load("D:\\EM0122\\Encoding\\Imaging\\30Hz\\suite2p\\plane0\\iscell.npy", allow_pickle=True)
Video = PreProcessing.loadRawBinary("", "", "D:\\EM0122\\Encoding\\Imaging\\30Hz\\denoised\\converted")[0:1000]
Stat = np.load(
    "D:\\EM0122\\Encoding\\Imaging\\30Hz\\suite2p\\plane0\\stat.npy", allow_pickle=True)
IsCell = np.load(
    "D:\\EM0122\\Encoding\\Imaging\\30Hz\\suite2P\\plane0\\iscell.npy", allow_pickle=True)
# neurons = np.where(IsCell[:, 0] == 1)[0].astype(int)

neurons = [0, 121, 267, 273, 315]

CV = colorize_video(Video, Stat, neurons, colors=[(1, 1, 1), (0.0745, 0.6234, 1.000),
                                                              (0.0745, 0.6234, 1.000)], write=True,
                    background=True, white_background=False, filename="C:\\Users\\YUSTE\\Desktop\\Test_11_11_22_03.mp4")

F1 = plt.figure(1)
A1 = F1.add_subplot(111)
A1.imshow(CV[0, :, :, :])
