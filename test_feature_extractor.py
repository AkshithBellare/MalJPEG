from struct import unpack
import pandas as pd
import glob

marker_mapping = {
   0xffc0: "SOF0",
   0xffc1: "SOF1",
   0xffc2: "SOF2",
   0xffc3: "SOF3",
   0xffc4: "DHT",
   0xffc5: "SOF5",
   0xffc6: "SOF6",
   0xffc7: "SOF7",
   0xffc8: "JPG",
   0xffc9: "SOF9",
   0xffca: "SOF10",
   0xffcb: "SOF11",
   0xffcc: "DAC",
   0xffcd: "SOF13",
   0xffce: "SOF14",
   0xffcf: "SOF15",
   0xffd0: "RST0",
   0xffd1: "RST1",
   0xffd2: "RST2",
   0xffd3: "RST3",
   0xffd4: "RST4",
   0xffd5: "RST5",
   0xffd6: "RST6",
   0xffd7: "RST7",
   0xffd8: "SOI",
   0xffd9: "EOI",
   0xffda: "SOS",
   0xffdb: "DQT",
   0xffdc: "DNL",
   0xffdd: "DRI",
   0xffde: "DHP",
   0xffdf: "EXP",
   0xffe0: "APP0",
   0xffe1: "APP1",
   0xffe2: "APP2",
   0xffe3: "APP3",
   0xffe4: "APP4",
   0xffe5: "APP5",
   0xffe6: "APP6",
   0xffe7: "APP7",
   0xffe8: "APP8",
   0xffe9: "APP9",
   0xffea: "APP10",
   0xffeb: "APP11",
   0xffec: "APP12",
   0xffed: "APP13",
   0xffee: "APP14",
   0xffef: "APP15",
   0xfff0: "JPG0",
   0xfff1: "JPG1",
   0xfff2: "JPG2",
   0xfff3: "JPG3",
   0xfff4: "JPG4",
   0xfff5: "JPG5",
   0xfff6: "JPG6",
   0xfff7: "JPG7",
   0xfff8: "JPG8",
   0xfff9: "JPG9",
   0xfffa: "JPG10",
   0xfffb: "JPG11",
   0xfffc: "JPG12",
   0xfffd: "JPG13",
   0xfffe: "COM",
   0xff01: "TEM",
}

class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        print(data[1])
        marker_DQT_num = 0
        marker_DQT_size_max = 0
        marker_DHT_num = 0
        marker_DHT_size_max = 0
        file_markers_num = 0
        marker_EOI_content_after_num = 0
        marker_APP12_size_max = 0
        marker_APP1_size_max = 0
        marker_COM_size_max = 0
        file_size = len(data)
        print(f"file_size = {file_size}")
        while(True):
            try:
                marker, = unpack(">H", data[0:2])
            except:
                print("error")
            marker_map = marker_mapping.get(marker)
            if marker_map != None:
                file_markers_num += 1
                print(marker_map)
                if marker_map == "DQT":
                    marker_DQT_num += 1
                    lenchunk, = unpack(">H", data[2:4])
                    if lenchunk > marker_DQT_size_max:
                        marker_DQT_size_max = lenchunk
                    data = data[2+lenchunk:]
                elif marker_map == "SOI":
                    data = data[2:]
                elif marker_map == "DHT":
                    marker_DHT_num += 1
                    lenchunk, = unpack(">H", data[2:4])
                    if lenchunk > marker_DHT_size_max:
                        marker_DHT_size_max = lenchunk
                    data = data[2+lenchunk:]
                elif marker_map == "EOI":
                    rem = data[2:]
                    if len(rem) > marker_EOI_content_after_num:
                        marker_EOI_content_after_num = len(rem)
                    data = rem
                elif marker_map == "SOS":
                    data = data[2:]
                elif marker_map == "APP12":
                    lenchunk, = unpack(">H", data[2:4])
                    if lenchunk > marker_APP12_size_max:
                        marker_APP12_size_max = lenchunk
                    data = data[2+lenchunk:]
                elif marker_map == "APP1":
                    lenchunk, = unpack(">H", data[2:4])
                    if lenchunk > marker_APP1_size_max:
                        marker_APP1_size_max = lenchunk
                    data = data[2+lenchunk:]
                elif marker_map == "COM":
                    lenchunk, = unpack(">H", data[2:4])
                    if lenchunk > marker_COM_size_max:
                        marker_COM_size_max = lenchunk
                    data = data[2+lenchunk:]
                elif marker_map == "TEM":
                    data = data[2:]
                elif marker <= 0xffd9 and marker >= 0xffd0:
                    data = data[2:]
                elif marker <= 0xffbf and marker >= 0xff02:
                    lenchunk, = unpack(">H", data[2:4])
                    data = data[2+lenchunk:]
                else:
                    lenchunk, = unpack(">H", data[2:4])
                    data = data[2+lenchunk:]
            else:
                data = data[1:]
            if (len(data) == 0):
                 data_list = [marker_EOI_content_after_num,marker_DQT_num,marker_DHT_num,file_markers_num, marker_DQT_size_max, marker_DHT_size_max,file_size, marker_COM_size_max,marker_APP1_size_max,marker_APP12_size_max,0]
               # print(marker_DQT_num)
               # print(marker_DHT_num)
               # print(file_markers_num)
               # print(marker_EOI_content_after_num)
               # print(marker_DQT_size_max)
               # print(marker_DHT_size_max)
               # print(file_size)
               # print(marker_COM_size_max)
               # print(marker_APP1_size_max)
               # print(marker_APP12_size_max)
                 return data_list 
          #  marker, = unpack(">H", data[0:2])
          #  print(marker_mapping.get(marker))
          #  print(hex(marker))
          #  if marker == 0xffd8:
          #      data = data[2:]
          #  elif marker == 0xffd9:
          #      return
          #  elif marker == 0xffda:
          #      data = data[-2:]
          #  else:
          #      lenchunk, = unpack(">H", data[2:4])
          #      data = data[2+lenchunk:]            
          #  if len(data)==0:
          #      break        

if __name__ == "__main__":
    path = r'E:\malicious' # ute your path i.e the folder with csv files from ecg viewer
    all_files = glob.glob(path + "/*.jpg")
    all_data = []
    for filename in all_files:
        img = JPEG(filename)
        data_list = img.decode()
        all_data.append(data_list)
        print(filename)
    print(all_data)
    #columns = ["marker_EOI_content_after_num","marker_DQT_num","marker_DHT_num","file_markers_num", "marker_DQT_size_max", "marker_DHT_size_max","file_size", "marker_COM_size_max","marker_APP1_size_max","marker_APP12_size_max", "target"])
    df = pd.DataFrame(all_data, columns = ["marker_EOI_content_after_num","marker_DQT_num","marker_DHT_num","file_markers_num", "marker_DQT_size_max", "marker_DHT_size_max","file_size", "marker_COM_size_max","marker_APP1_size_max","marker_APP12_size_max", "target"])
    #input_df = pd.concat(all_data, axis=0, ignore_index=True)
    df.to_csv('malicious_features.csv', mode="w",index=False)
# OUTPUT:
# Start of Image
# Application Default Header
# Quantization Table
# Quantization Table
# Start of Frame
# Huffman Table
# Huffman Table
# Huffman Table
# Huffman Table
# Start of Scan
# End of Image