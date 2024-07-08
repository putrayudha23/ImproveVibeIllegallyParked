import numpy as np
import cv2
from skimage import feature
import pickle

class vibe_gray:
    def __init__(self, ):
        self.width = 0
        self.height = 0
        self.numberOfSamples = 20 # default 20 = N (jumlah sample)
        self.matchingThreshold = 20 # default 20 = R (nilai radius)
        self.matchingNumber = 2 # default 2 = #min (cardinality min) threshold u/ menentukan foreground/background
        self.updateFactor = 16 # default 16 = φ (time subsampling factor = probabilitas 1/16 untuk update) kalo besar long term, kecil sort term
        
        # Storage for the history
        self.historyImage = None # modelnya ada disini
        self.historyBuffer = None # modelnya ada disini
        self.lastHistoryImageSwapped = 0
        self.numberOfHistoryImages = 2
        
        # Buffers with random values
        self.jump = None
        self.neighbor = None
        self.position = None
        
    def AllocInit(self, image):
        print('AllocInit!')
        height, width = image.shape[:2] #simpan height dan width dari image
        # set the parameters
        self.width = width
        self.height = height
        print(self.height, self.width)
        
        # create the historyImage
        self.historyImage = np.zeros((self.height, self.width, self.numberOfHistoryImages), np.uint8) # ada (height) matriks 0 dengan ukuran (width*numberOfHistoryImages) didalam suatu matriks. uint = unsigned long
        for i in range(self.numberOfHistoryImages):
            self.historyImage[:, :, i] = image # setiap matriks ukuran image (height*width). i sebanyak (numberOfHistoryImages = 2) = i:0, i:1
            
        # create and fill the historyBuffer
        # buat array dengan 18 matriks frame 0 didalamnya. 18 didapat dari numberOfSamples-self.numberOfHistoryImages=20-2=18 (i = 0 sd 17)
        self.historyBuffer = np.zeros((self.height, self.width, self.numberOfSamples-self.numberOfHistoryImages), np.uint8)
        for i in range(self.numberOfSamples-self.numberOfHistoryImages): #20-2 = 18 (0-17)
            image_plus_noise = image + np.random.randint(-10, 10, (self.height, self.width)) #image + random -10 sd 9 di simpan di matriks height x width (ukuran image)
            image_plus_noise[image_plus_noise > 255] = 255 #yang nilainya lebih besar dari 225 di image_plus_noise diset menjadi 255
            image_plus_noise[image_plus_noise < 0] = 0 #yang nilainya lebih kecil dari 0 di image_plus_noise diset menjadi 0
            self.historyBuffer[:, :, i] = image_plus_noise.astype(np.uint8) # historyBuffer diisi image_plus_noise
        
        # fill the buffers with random values (jump, neighbor, position)
        size = 2 * self.width + 1 if (self.width > self.height) else 2 * self.height + 1
        self.jump = np.zeros((size), np.uint32) # array 0 dengan banyak anggota size [0,0,0,0,...] (long 32)
        self.neighbor = np.zeros((size), np.int) # array 0 dengan banyak anggota size [0,0,0,0,...] (integer)
        self.position = np.zeros((size), np.uint32) # array 0 dengan banyak anggota size [0,0,0,0,...] (long 32)
        for i in range(size):
            self.jump[i] = np.random.randint(1, 2*self.updateFactor+1) #nilai int random 1-33 (1...32)
            self.neighbor[i] = np.random.randint(-1, 3-1) + np.random.randint(-1, 3-1) * self.width #nilai random int (-1..1) + nilai random int (-1..1) * width
            self.position[i] = np.random.randint(0, self.numberOfSamples) #nilai random (0..19)       
    
    def Segmentation(self, image):
        # segmentation_map init
        segmentation_map = np.zeros((self.height, self.width)) + (self.matchingNumber - 1) #hasinya matriks 1 height x width
        
        # first history image # BACKGROUND MODEL AWAL (segmentation_map) DIPENGARUHI OLEH NILAI historyImage -> mask
        mask = np.abs(image - self.historyImage[:, :, 0]) > self.matchingThreshold # hasilnya (matriks berisi true false) dgn kondisi (image-historyImage yang pertama > R=20)
        segmentation_map[mask] = self.matchingNumber # matriks 1 (segmentation_map) dimodifikasi berapa nilainya dengan #min=2 yang sesuai mask

        # next historyImages
        for i in range(1, self.numberOfHistoryImages): # i di range(1,2) = 1
            mask = np.abs(image - self.historyImage[:, :, i]) <= self.matchingThreshold # hasilnya (matriks berisi true false) dgn kondisi (image - historyImage ke 1 (tidak ke 0 lagi karena ud next) > R=20)
            segmentation_map[mask] = segmentation_map[mask] - 1 # matriks 1 (segmentation_map) dimodifikasi berapa nilainya dengan -1 yang sesuai mask >> matriks 1 yang berisi anggota 0 dan 1 (1 untuk yg sebelumnya bernilai 2)
        
        # for swapping
        self.lastHistoryImageSwapped = (self.lastHistoryImageSwapped + 1) % self.numberOfHistoryImages #operasi modulo iterasi pertama (1 mod 2) = 1. sisa 1 (tiap iterasi nilai akan 1 0 1 0 1 0 1 ....)
        swappingImageBuffer = self.historyImage[:, :, self.lastHistoryImageSwapped] #swappingImageBuffer = historyimage 1 [1 0 1 0 1 0 ....]
        
        # now, we move in the buffer and leave the historyImage
        numberOfTests = self.numberOfSamples - self.numberOfHistoryImages #20-2=18
        mask = segmentation_map > 0 # matriks true false
        for i in range(numberOfTests): # 0-17
            mask_ = np.abs(image - self.historyBuffer[:, :, i]) <= self.matchingThreshold #<=20 [menghasilkan matriks true false]
            mask_ = mask * mask_ # menghasilkan matriks 0 1
            segmentation_map[mask_] = segmentation_map[mask_] - 1 # beberapa nilai 1 jadi 0. nilai 0 jadi -1 (matriks 1 0 -1)
            
            # Swapping: Putting found value in history image buffer
            temp = swappingImageBuffer[mask_].copy()
            swappingImageBuffer[mask_] = self.historyBuffer[:, :, i][mask_].copy()
            self.historyBuffer[:, :, i][mask_] = temp #history buffer 0-17 diisi
        
        # simulate the exit inner loop
        mask_ = segmentation_map <= 0
        mask_ = mask * mask_
        segmentation_map[mask_] = 0
        
        # Produces the output. Note that this step is application-dependent
        mask = segmentation_map > 0 # foreground
        segmentation_map[mask] = 255 # foreground diset warnanya jadi putih

        # UNTUK BACKGROUND TIDAK PERLU SET 0 LAGI KARENA ASALNYA EMG SUDAH 0
        #mask2 = segmentation_map <= 0 # background
        #segmentation_map[mask2] = 0 # background diset warnanya jadi hitam
        
        return segmentation_map.astype(np.uint8)
    
    def UpdateLess20pxl(self, image, updating_mask):
        # hitung jumlah piksel foreground
        ret,thresh = cv2.threshold(updating_mask,127,255,cv2.THRESH_BINARY)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        i = 1
        
        # Update background jika piksel blob < 20
        while i > 0 and i <= n-1: #loop per blob
            numPixel = stats[i, 4] # ambil nilai jumlah pikselnya # [i,4]

            if numPixel < 20 : # < 20 (jika foreground/blob memiliki jumlah piksel < 20 maka akan dijadikan background pada frame berikutnya)
                shift = np.random.randint(0, self.width)
                # ambil koordinatnya tiap piksel pada bbox blop # ukuran gambar max [480,640]
                for y in range(stats[i,1], stats[i,1]+stats[i,3]-1):
                    for x in range(stats[i,0], stats[i,0]+stats[i,2]-1):
                        # Udate background
                        if updating_mask[y,x] == 255:
                            value = image[y, x]
                            if self.position[shift] < self.numberOfHistoryImages:
                                self.historyImage[y, x, self.position[shift]] = value
                            else:
                                pos = self.position[shift] - self.numberOfHistoryImages
                                self.historyBuffer[y, x, pos] = value

            i = i+1

    def UpdateLess20pxlSVM(self, image, updating_mask):
        # hitung jumlah piksel foreground
        ret,thresh = cv2.threshold(updating_mask,127,255,cv2.THRESH_BINARY)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        i = 1

        #simpan koordinat bbox
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0

        # load model SVM
        with open('modelSVM_Gray144.pkl', 'rb') as f: #model2Gray144.pkl
            svm_model = pickle.load(f)
        
        # Update background jika piksel blob < 20
        while i > 0 and i <= n-1: #loop per blob
            numPixel = stats[i, 4] # ambil nilai jumlah pikselnya # [i,4]

            if numPixel < 20 : # < 20 (jika foreground/blob memiliki jumlah piksel < 20 maka akan dijadikan background pada frame berikutnya)
                shift = np.random.randint(0, self.width)
                # ambil koordinatnya tiap piksel pada bbox blop # ukuran gambar max [480,640]
                for y in range(stats[i,1], stats[i,1]+stats[i,3]-1):
                    for x in range(stats[i,0], stats[i,0]+stats[i,2]-1):
                        # Udate background
                        if updating_mask[y,x] == 255:
                            value = image[y, x]
                            if self.position[shift] < self.numberOfHistoryImages:
                                self.historyImage[y, x, self.position[shift]] = value
                            else:
                                pos = self.position[shift] - self.numberOfHistoryImages
                                self.historyBuffer[y, x, pos] = value
            else: # jika > 20 piksel
                #========================================
                baris1 = stats[i,1]
                baris2 = stats[i,1]+stats[i,3]-1
                kolom1 = stats[i,0]
                kolom2 = stats[i,0]+stats[i,2]-1

                if baris1==baris2:
                    object = image[baris1:baris2+1, kolom1:kolom2]
                elif kolom1==kolom2:
                    object = image[baris1:baris2, kolom1:kolom2+1]
                elif baris1==baris2 and kolom1==kolom2:
                    object = image[baris1:baris2+1, kolom1:kolom2+1]
                else:
                    object = image[baris1:baris2,kolom1:kolom2]

                resized_image = cv2.resize(object, (144, 144)) # ukuran dari model SVM yang telah digenerate

                # generate fitur HOG
                (hog_desc, hog_image) = feature.hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)

                pred = svm_model.predict(hog_desc.reshape(1, -1))[0]

                if pred != 'car':
                    shift = np.random.randint(0, self.width)
                    for y in range(stats[i,1], stats[i,1]+stats[i,3]-1):
                        for x in range(stats[i,0], stats[i,0]+stats[i,2]-1):
                            # Udate background
                            if updating_mask[y,x] == 255:
                                value = image[y, x]
                                if self.position[shift] < self.numberOfHistoryImages:
                                    self.historyImage[y, x, self.position[shift]] = value
                                else:
                                    pos = self.position[shift] - self.numberOfHistoryImages
                                    self.historyBuffer[y, x, pos] = value
                else:
                    # koordinat bounding box
                    x1 = stats[i,0]
                    y1 = stats[i,1]
                    x2 = stats[i,0]+stats[i,2]
                    y2 = stats[i,1]+stats[i,3]
                #========================================

            i = i+1
        
        return x1, y1, x2, y2

