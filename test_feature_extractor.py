#MARKER = {
#    0xFFC0: ("SOF0", "Baseline DCT", SOF),
#    0xFFC1: ("SOF1", "Extended Sequential DCT", SOF),
#    0xFFC2: ("SOF2", "Progressive DCT", SOF),
#    0xFFC3: ("SOF3", "Spatial lossless", SOF),
#    0xFFC4: ("DHT", "Define Huffman table", Skip),
#    0xFFC5: ("SOF5", "Differential sequential DCT", SOF),
#    0xFFC6: ("SOF6", "Differential progressive DCT", SOF),
#    0xFFC7: ("SOF7", "Differential spatial", SOF),
#    0xFFC8: ("JPG", "Extension", None),
#    0xFFC9: ("SOF9", "Extended sequential DCT (AC)", SOF),
#    0xFFCA: ("SOF10", "Progressive DCT (AC)", SOF),
#    0xFFCB: ("SOF11", "Spatial lossless DCT (AC)", SOF),
#    0xFFCC: ("DAC", "Define arithmetic coding conditioning", Skip),
#    0xFFCD: ("SOF13", "Differential sequential DCT (AC)", SOF),
#    0xFFCE: ("SOF14", "Differential progressive DCT (AC)", SOF),
#    0xFFCF: ("SOF15", "Differential spatial (AC)", SOF),
#    0xFFD0: ("RST0", "Restart 0", None),
#    0xFFD1: ("RST1", "Restart 1", None),
#    0xFFD2: ("RST2", "Restart 2", None),
#    0xFFD3: ("RST3", "Restart 3", None),
#    0xFFD4: ("RST4", "Restart 4", None),
#    0xFFD5: ("RST5", "Restart 5", None),
#    0xFFD6: ("RST6", "Restart 6", None),
#    0xFFD7: ("RST7", "Restart 7", None),
#    0xFFD8: ("SOI", "Start of image", None),
#    0xFFD9: ("EOI", "End of image", None),
#    0xFFDA: ("SOS", "Start of scan", Skip),
#    0xFFDB: ("DQT", "Define quantization table", DQT),
#    0xFFDC: ("DNL", "Define number of lines", Skip),
#    0xFFDD: ("DRI", "Define restart interval", Skip),
#    0xFFDE: ("DHP", "Define hierarchical progression", SOF),
#    0xFFDF: ("EXP", "Expand reference component", Skip),
#    0xFFE0: ("APP0", "Application segment 0", APP),
#    0xFFE1: ("APP1", "Application segment 1", APP),
#    0xFFE2: ("APP2", "Application segment 2", APP),
#    0xFFE3: ("APP3", "Application segment 3", APP),
#    0xFFE4: ("APP4", "Application segment 4", APP),
#    0xFFE5: ("APP5", "Application segment 5", APP),
#    0xFFE6: ("APP6", "Application segment 6", APP),
#    0xFFE7: ("APP7", "Application segment 7", APP),
#    0xFFE8: ("APP8", "Application segment 8", APP),
#    0xFFE9: ("APP9", "Application segment 9", APP),
#    0xFFEA: ("APP10", "Application segment 10", APP),
#    0xFFEB: ("APP11", "Application segment 11", APP),
#    0xFFEC: ("APP12", "Application segment 12", APP),
#    0xFFED: ("APP13", "Application segment 13", APP),
#    0xFFEE: ("APP14", "Application segment 14", APP),
#    0xFFEF: ("APP15", "Application segment 15", APP),
#    0xFFF0: ("JPG0", "Extension 0", None),
#    0xFFF1: ("JPG1", "Extension 1", None),
#    0xFFF2: ("JPG2", "Extension 2", None),
#    0xFFF3: ("JPG3", "Extension 3", None),
#    0xFFF4: ("JPG4", "Extension 4", None),
#    0xFFF5: ("JPG5", "Extension 5", None),
#    0xFFF6: ("JPG6", "Extension 6", None),
#    0xFFF7: ("JPG7", "Extension 7", None),
#    0xFFF8: ("JPG8", "Extension 8", None),
#    0xFFF9: ("JPG9", "Extension 9", None),
#    0xFFFA: ("JPG10", "Extension 10", None),
#    0xFFFB: ("JPG11", "Extension 11", None),
#    0xFFFC: ("JPG12", "Extension 12", None),
#    0xFFFD: ("JPG13", "Extension 13", None),
#    0xFFFE: ("COM", "Comment", COM)
#}

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
                    data = b"\xff"
                    print(data)
                    print(len(data))
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
                 return data_list 

if __name__ == "__main__":
    path = r'/home/axebell/Desktop' # ute your path i.e the folder with csv files from ecg viewer
    all_files = glob.glob(path + "/*.jpg")
    all_data = []
    img_count = 0
    for filename in all_files:
        img_count += 1
        img = JPEG(filename)
        print(f"filename = {filename} img_count = {img_count}")
        data_list = img.decode()
        all_data.append(data_list)
    print(all_data)
    df = pd.DataFrame(all_data, columns = ["marker_EOI_content_after_num","marker_DQT_num","marker_DHT_num","file_markers_num", "marker_DQT_size_max", "marker_DHT_size_max","file_size", "marker_COM_size_max","marker_APP1_size_max","marker_APP12_size_max", "target"])
    df.to_csv('test.csv', mode="w",index=False)