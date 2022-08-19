
import torch
from torch import tensor, device
import torch.fx as fx
from torchdynamo.testing import rand_strided
from math import inf
from torchinductor.compile_fx import compile_fx_inner
from torch.fx.experimental.proxy_tensor import make_fx

class Repro(torch.nn.Module):



    def forward(self, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg188_1, arg189_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg224_1, arg225_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg260_1, arg261_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg327_1, arg328_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg363_1, arg364_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg399_1, arg400_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg435_1, arg436_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg471_1, arg472_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg507_1, arg508_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg976_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1018_1, arg1019_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1060_1, arg1061_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1102_1, arg1103_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1194_1, arg1195_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1, arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1236_1, arg1237_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1278_1, arg1279_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1320_1, arg1321_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1362_1, arg1363_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1, arg1380_1, arg1381_1, arg1382_1, arg1383_1, arg1384_1, arg1385_1, arg1386_1, arg1387_1, arg1388_1, arg1389_1, arg1390_1, arg1391_1, arg1392_1, arg1393_1, arg1394_1, arg1395_1, arg1396_1, arg1397_1, arg1398_1, arg1399_1, arg1400_1, arg1401_1, arg1404_1, arg1405_1, arg1408_1, arg1409_1, arg1410_1, arg1411_1, arg1412_1, arg1413_1, arg1414_1, arg1415_1, arg1416_1, arg1417_1, arg1418_1, arg1419_1, arg1420_1, arg1421_1, arg1422_1, arg1423_1, arg1424_1, arg1425_1, arg1426_1, arg1427_1, arg1428_1, arg1429_1, arg1430_1, arg1431_1, arg1432_1, arg1433_1, arg1434_1, arg1435_1, arg1436_1, arg1437_1, arg1438_1, arg1439_1, arg1440_1, arg1441_1, arg1442_1, arg1443_1, arg1444_1, arg1445_1, arg1446_1, arg1447_1, arg1448_1, arg1449_1, arg1450_1, arg1451_1, arg1452_1, arg1453_1, arg1454_1, arg1455_1, arg1456_1, arg1457_1, arg1458_1, arg1459_1, arg1460_1, arg1461_1, arg1462_1, arg1463_1, arg1464_1, arg1465_1, arg1466_1, arg1467_1, arg1468_1, arg1469_1, arg1470_1, arg1471_1, arg1472_1, arg1473_1, arg1474_1, arg1475_1, arg1476_1, arg1477_1, arg1478_1, arg1479_1, arg1480_1, arg1481_1, arg1482_1, arg1483_1, arg1484_1, arg1485_1, arg1486_1, arg1487_1, arg1489_1, arg1809_1, arg1810_1, arg1811_1, arg1812_1, arg1813_1, arg1814_1, arg1815_1, arg1816_1, arg1817_1, arg1818_1, arg1819_1, arg1820_1, arg1821_1, arg1822_1, arg1823_1, arg1824_1, arg1825_1, arg1826_1, arg1827_1, arg1828_1, arg1829_1, arg1830_1, arg1831_1, arg1832_1, arg1833_1, arg1834_1, arg1835_1, arg1836_1, arg1837_1, arg1838_1, arg1839_1, arg1840_1, arg1841_1, arg1842_1, arg1843_1, arg1844_1, arg1845_1, arg1846_1, arg1847_1, arg1848_1, arg1849_1, arg1850_1, arg1851_1, arg1852_1, arg1853_1, arg1854_1, arg1855_1, arg1856_1, arg1857_1, arg1858_1, arg1859_1, arg1860_1, arg1861_1, arg1862_1, arg1863_1, arg1864_1, arg1865_1, arg1866_1, arg1867_1, arg1868_1, arg1869_1, arg1870_1, arg1871_1, arg1872_1, arg1873_1, arg1874_1, arg1875_1, arg1876_1, arg1877_1, arg1878_1, arg1879_1, arg1880_1, arg1881_1, arg1882_1, arg1883_1, arg1884_1, arg1885_1, arg1886_1, arg1887_1, arg1888_1, arg1889_1, arg1890_1, arg1891_1, arg1892_1, arg1893_1, arg1894_1, arg1895_1, arg1896_1, arg1897_1, arg1898_1, arg1899_1, arg1900_1, arg1901_1, arg1902_1, arg1903_1, arg1904_1, arg1905_1, arg1906_1, arg1907_1, arg1908_1, arg1909_1, arg1910_1, arg1911_1, arg1912_1, arg1913_1, arg1914_1, arg1915_1, arg1916_1, arg1917_1, arg1918_1, arg1919_1, arg1920_1, arg1921_1, arg1922_1, arg1923_1, arg1924_1, arg1925_1, arg1926_1, arg1927_1, arg1928_1, arg1929_1, arg1930_1, arg1931_1, arg1932_1, arg1933_1, arg1934_1, arg1935_1, add_tensor_4, add_tensor_5, relu_default_6, relu_default_7, squeeze_dim_38, squeeze_dim_41, add_tensor_6, squeeze_dim_44, squeeze_dim_47, add_tensor_7, relu_default_8, relu_default_9, squeeze_dim_50, squeeze_dim_53, add_tensor_8, squeeze_dim_56, squeeze_dim_59, add_tensor_9, relu_default_10, relu_default_11, squeeze_dim_62, squeeze_dim_65, add_tensor_10, squeeze_dim_68, squeeze_dim_71, add_tensor_11, relu_default_12, relu_default_13, relu_default_14, squeeze_dim_74, squeeze_dim_77, add_tensor_12, squeeze_dim_80, squeeze_dim_83, add_tensor_13, relu_default_15, relu_default_16, squeeze_dim_86, squeeze_dim_89, add_tensor_14, squeeze_dim_92, squeeze_dim_95, add_tensor_15, relu_default_17, relu_default_18, squeeze_dim_98, squeeze_dim_101, add_tensor_16, squeeze_dim_104, squeeze_dim_107, add_tensor_17, relu_default_19, relu_default_20, squeeze_dim_110, squeeze_dim_113, add_tensor_18, squeeze_dim_116, squeeze_dim_119, add_tensor_19, relu_default_21, relu_default_22, squeeze_dim_122, squeeze_dim_125, add_tensor_20, squeeze_dim_128, squeeze_dim_131, add_tensor_21, relu_default_23, relu_default_24, squeeze_dim_134, squeeze_dim_137, add_tensor_22, squeeze_dim_140, squeeze_dim_143, add_tensor_23, relu_default_25, relu_default_26, relu_default_27, le_scalar_48, where_self_72, slice_tensor_38, slice_tensor_39, slice_tensor_40, slice_tensor_41, constant_pad_nd_default_1, sum_dim_int_list_132, sub_tensor_252):
        mul_tensor_708 = torch.ops.aten.mul.Tensor(slice_tensor_41, sub_tensor_252)
        sum_dim_int_list_133 = torch.ops.aten.sum.dim_IntList(mul_tensor_708, [0, 2, 3]);  mul_tensor_708 = None
        mul_tensor_709 = torch.ops.aten.mul.Tensor(sum_dim_int_list_132, 0.008264462809917356);  sum_dim_int_list_132 = None
        view_default_217 = torch.ops.aten.view.default(mul_tensor_709, [1, 672, 1, 1]);  mul_tensor_709 = None
        mul_tensor_710 = torch.ops.aten.mul.Tensor(sum_dim_int_list_133, 0.008264462809917356)
        mul_tensor_711 = torch.ops.aten.mul.Tensor(arg1489_1, arg1489_1)
        mul_tensor_712 = torch.ops.aten.mul.Tensor(mul_tensor_710, mul_tensor_711);  mul_tensor_710 = mul_tensor_711 = None
        view_default_218 = torch.ops.aten.view.default(mul_tensor_712, [1, 672, 1, 1]);  mul_tensor_712 = None
        mul_tensor_713 = torch.ops.aten.mul.Tensor(arg1489_1, arg569_1);  arg569_1 = None
        view_default_219 = torch.ops.aten.view.default(mul_tensor_713, [1, 672, 1, 1]);  mul_tensor_713 = None
        mul_tensor_714 = torch.ops.aten.mul.Tensor(sub_tensor_252, view_default_218);  sub_tensor_252 = view_default_218 = None
        sub_tensor_253 = torch.ops.aten.sub.Tensor(slice_tensor_41, mul_tensor_714);  slice_tensor_41 = mul_tensor_714 = None
        sub_tensor_254 = torch.ops.aten.sub.Tensor(sub_tensor_253, view_default_217);  sub_tensor_253 = view_default_217 = None
        mul_tensor_715 = torch.ops.aten.mul.Tensor(sub_tensor_254, view_default_219);  sub_tensor_254 = view_default_219 = None
        mul_tensor_716 = torch.ops.aten.mul.Tensor(sum_dim_int_list_133, arg1489_1);  sum_dim_int_list_133 = arg1489_1 = None
        convolution_backward_default_133 = torch.ops.aten.convolution_backward.default(mul_tensor_715, arg1487_1, arg568_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_715 = arg1487_1 = arg568_1 = None
        getitem_399 = convolution_backward_default_133[0]
        getitem_400 = convolution_backward_default_133[1];  convolution_backward_default_133 = None
        convolution_backward_default_134 = torch.ops.aten.convolution_backward.default(getitem_399, arg1486_1, arg567_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 672, [True, True, False]);  getitem_399 = arg567_1 = None
        getitem_402 = convolution_backward_default_134[0]
        getitem_403 = convolution_backward_default_134[1];  convolution_backward_default_134 = None
        le_scalar_49 = torch.ops.aten.le.Scalar(arg1486_1, 0);  arg1486_1 = None
        new_zeros_default_73 = torch.ops.aten.new_zeros.default(getitem_402, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_73 = torch.ops.aten.where.self(le_scalar_49, new_zeros_default_73, getitem_402);  le_scalar_49 = new_zeros_default_73 = getitem_402 = None
        sum_dim_int_list_134 = torch.ops.aten.sum.dim_IntList(where_self_73, [0, 2, 3])
        sub_tensor_255 = torch.ops.aten.sub.Tensor(arg1484_1, arg1809_1);  arg1484_1 = arg1809_1 = None
        mul_tensor_717 = torch.ops.aten.mul.Tensor(where_self_73, sub_tensor_255)
        sum_dim_int_list_135 = torch.ops.aten.sum.dim_IntList(mul_tensor_717, [0, 2, 3]);  mul_tensor_717 = None
        mul_tensor_718 = torch.ops.aten.mul.Tensor(sum_dim_int_list_134, 0.008264462809917356);  sum_dim_int_list_134 = None
        view_default_220 = torch.ops.aten.view.default(mul_tensor_718, [1, 672, 1, 1]);  mul_tensor_718 = None
        mul_tensor_719 = torch.ops.aten.mul.Tensor(sum_dim_int_list_135, 0.008264462809917356)
        mul_tensor_720 = torch.ops.aten.mul.Tensor(arg1485_1, arg1485_1)
        mul_tensor_721 = torch.ops.aten.mul.Tensor(mul_tensor_719, mul_tensor_720);  mul_tensor_719 = mul_tensor_720 = None
        view_default_221 = torch.ops.aten.view.default(mul_tensor_721, [1, 672, 1, 1]);  mul_tensor_721 = None
        mul_tensor_722 = torch.ops.aten.mul.Tensor(arg1485_1, arg566_1);  arg566_1 = None
        view_default_222 = torch.ops.aten.view.default(mul_tensor_722, [1, 672, 1, 1]);  mul_tensor_722 = None
        mul_tensor_723 = torch.ops.aten.mul.Tensor(sub_tensor_255, view_default_221);  sub_tensor_255 = view_default_221 = None
        sub_tensor_256 = torch.ops.aten.sub.Tensor(where_self_73, mul_tensor_723);  where_self_73 = mul_tensor_723 = None
        sub_tensor_257 = torch.ops.aten.sub.Tensor(sub_tensor_256, view_default_220);  sub_tensor_256 = view_default_220 = None
        mul_tensor_724 = torch.ops.aten.mul.Tensor(sub_tensor_257, view_default_222);  sub_tensor_257 = view_default_222 = None
        mul_tensor_725 = torch.ops.aten.mul.Tensor(sum_dim_int_list_135, arg1485_1);  sum_dim_int_list_135 = arg1485_1 = None
        convolution_backward_default_135 = torch.ops.aten.convolution_backward.default(mul_tensor_724, arg1483_1, arg565_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_724 = arg1483_1 = arg565_1 = None
        getitem_405 = convolution_backward_default_135[0]
        getitem_406 = convolution_backward_default_135[1];  convolution_backward_default_135 = None
        convolution_backward_default_136 = torch.ops.aten.convolution_backward.default(getitem_405, relu_default_27, arg564_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 672, [True, True, False]);  getitem_405 = arg564_1 = None
        getitem_408 = convolution_backward_default_136[0]
        getitem_409 = convolution_backward_default_136[1];  convolution_backward_default_136 = None
        le_scalar_50 = torch.ops.aten.le.Scalar(relu_default_27, 0);  relu_default_27 = None
        new_zeros_default_74 = torch.ops.aten.new_zeros.default(getitem_408, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_74 = torch.ops.aten.where.self(le_scalar_50, new_zeros_default_74, getitem_408);  le_scalar_50 = new_zeros_default_74 = getitem_408 = None
        add_tensor_96 = torch.ops.aten.add.Tensor(slice_tensor_38, slice_tensor_40);  slice_tensor_38 = None
        avg_pool2d_backward_default_14 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_40, arg1464_1, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_40 = arg1464_1 = None
        add_tensor_97 = torch.ops.aten.add.Tensor(where_self_74, avg_pool2d_backward_default_14);  where_self_74 = avg_pool2d_backward_default_14 = None
        sum_dim_int_list_136 = torch.ops.aten.sum.dim_IntList(slice_tensor_39, [0, 2, 3])
        sub_tensor_258 = torch.ops.aten.sub.Tensor(arg1481_1, arg1810_1);  arg1481_1 = arg1810_1 = None
        mul_tensor_726 = torch.ops.aten.mul.Tensor(slice_tensor_39, sub_tensor_258)
        sum_dim_int_list_137 = torch.ops.aten.sum.dim_IntList(mul_tensor_726, [0, 2, 3]);  mul_tensor_726 = None
        mul_tensor_727 = torch.ops.aten.mul.Tensor(sum_dim_int_list_136, 0.008264462809917356);  sum_dim_int_list_136 = None
        view_default_223 = torch.ops.aten.view.default(mul_tensor_727, [1, 672, 1, 1]);  mul_tensor_727 = None
        mul_tensor_728 = torch.ops.aten.mul.Tensor(sum_dim_int_list_137, 0.008264462809917356)
        mul_tensor_729 = torch.ops.aten.mul.Tensor(arg1482_1, arg1482_1)
        mul_tensor_730 = torch.ops.aten.mul.Tensor(mul_tensor_728, mul_tensor_729);  mul_tensor_728 = mul_tensor_729 = None
        view_default_224 = torch.ops.aten.view.default(mul_tensor_730, [1, 672, 1, 1]);  mul_tensor_730 = None
        mul_tensor_731 = torch.ops.aten.mul.Tensor(arg1482_1, arg563_1);  arg563_1 = None
        view_default_225 = torch.ops.aten.view.default(mul_tensor_731, [1, 672, 1, 1]);  mul_tensor_731 = None
        mul_tensor_732 = torch.ops.aten.mul.Tensor(sub_tensor_258, view_default_224);  sub_tensor_258 = view_default_224 = None
        sub_tensor_259 = torch.ops.aten.sub.Tensor(slice_tensor_39, mul_tensor_732);  mul_tensor_732 = None
        sub_tensor_260 = torch.ops.aten.sub.Tensor(sub_tensor_259, view_default_223);  sub_tensor_259 = view_default_223 = None
        mul_tensor_733 = torch.ops.aten.mul.Tensor(sub_tensor_260, view_default_225);  sub_tensor_260 = view_default_225 = None
        mul_tensor_734 = torch.ops.aten.mul.Tensor(sum_dim_int_list_137, arg1482_1);  sum_dim_int_list_137 = arg1482_1 = None
        convolution_backward_default_137 = torch.ops.aten.convolution_backward.default(mul_tensor_733, arg1480_1, arg562_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_733 = arg1480_1 = arg562_1 = None
        getitem_411 = convolution_backward_default_137[0]
        getitem_412 = convolution_backward_default_137[1];  convolution_backward_default_137 = None
        convolution_backward_default_138 = torch.ops.aten.convolution_backward.default(getitem_411, arg1479_1, arg561_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  getitem_411 = arg561_1 = None
        getitem_414 = convolution_backward_default_138[0]
        getitem_415 = convolution_backward_default_138[1];  convolution_backward_default_138 = None
        le_scalar_51 = torch.ops.aten.le.Scalar(arg1479_1, 0);  arg1479_1 = None
        new_zeros_default_75 = torch.ops.aten.new_zeros.default(getitem_414, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_75 = torch.ops.aten.where.self(le_scalar_51, new_zeros_default_75, getitem_414);  le_scalar_51 = new_zeros_default_75 = getitem_414 = None
        sum_dim_int_list_138 = torch.ops.aten.sum.dim_IntList(where_self_75, [0, 2, 3])
        sub_tensor_261 = torch.ops.aten.sub.Tensor(arg1477_1, arg1811_1);  arg1477_1 = arg1811_1 = None
        mul_tensor_735 = torch.ops.aten.mul.Tensor(where_self_75, sub_tensor_261)
        sum_dim_int_list_139 = torch.ops.aten.sum.dim_IntList(mul_tensor_735, [0, 2, 3]);  mul_tensor_735 = None
        mul_tensor_736 = torch.ops.aten.mul.Tensor(sum_dim_int_list_138, 0.008264462809917356);  sum_dim_int_list_138 = None
        view_default_226 = torch.ops.aten.view.default(mul_tensor_736, [1, 672, 1, 1]);  mul_tensor_736 = None
        mul_tensor_737 = torch.ops.aten.mul.Tensor(sum_dim_int_list_139, 0.008264462809917356)
        mul_tensor_738 = torch.ops.aten.mul.Tensor(arg1478_1, arg1478_1)
        mul_tensor_739 = torch.ops.aten.mul.Tensor(mul_tensor_737, mul_tensor_738);  mul_tensor_737 = mul_tensor_738 = None
        view_default_227 = torch.ops.aten.view.default(mul_tensor_739, [1, 672, 1, 1]);  mul_tensor_739 = None
        mul_tensor_740 = torch.ops.aten.mul.Tensor(arg1478_1, arg560_1);  arg560_1 = None
        view_default_228 = torch.ops.aten.view.default(mul_tensor_740, [1, 672, 1, 1]);  mul_tensor_740 = None
        mul_tensor_741 = torch.ops.aten.mul.Tensor(sub_tensor_261, view_default_227);  sub_tensor_261 = view_default_227 = None
        sub_tensor_262 = torch.ops.aten.sub.Tensor(where_self_75, mul_tensor_741);  where_self_75 = mul_tensor_741 = None
        sub_tensor_263 = torch.ops.aten.sub.Tensor(sub_tensor_262, view_default_226);  sub_tensor_262 = view_default_226 = None
        mul_tensor_742 = torch.ops.aten.mul.Tensor(sub_tensor_263, view_default_228);  sub_tensor_263 = view_default_228 = None
        mul_tensor_743 = torch.ops.aten.mul.Tensor(sum_dim_int_list_139, arg1478_1);  sum_dim_int_list_139 = arg1478_1 = None
        convolution_backward_default_139 = torch.ops.aten.convolution_backward.default(mul_tensor_742, arg1476_1, arg559_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_742 = arg1476_1 = arg559_1 = None
        getitem_417 = convolution_backward_default_139[0]
        getitem_418 = convolution_backward_default_139[1];  convolution_backward_default_139 = None
        convolution_backward_default_140 = torch.ops.aten.convolution_backward.default(getitem_417, arg1475_1, arg15_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 672, [True, True, False]);  getitem_417 = arg1475_1 = arg15_1 = None
        getitem_420 = convolution_backward_default_140[0]
        getitem_421 = convolution_backward_default_140[1];  convolution_backward_default_140 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(getitem_420, [-2, -2, -2, -2]);  getitem_420 = None
        new_zeros_default_76 = torch.ops.aten.new_zeros.default(constant_pad_nd_default_2, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_76 = torch.ops.aten.where.self(arg1812_1, new_zeros_default_76, constant_pad_nd_default_2);  new_zeros_default_76 = constant_pad_nd_default_2 = None
        avg_pool2d_backward_default_15 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_39, arg1474_1, [3, 3], [2, 2], [0, 0], False, False, None);  slice_tensor_39 = arg1474_1 = None
        constant_pad_nd_default_3 = torch.ops.aten.constant_pad_nd.default(avg_pool2d_backward_default_15, [-1, -1, -1, -1]);  avg_pool2d_backward_default_15 = None
        add_tensor_98 = torch.ops.aten.add.Tensor(constant_pad_nd_default_1, constant_pad_nd_default_3);  constant_pad_nd_default_1 = constant_pad_nd_default_3 = None
        sum_dim_int_list_140 = torch.ops.aten.sum.dim_IntList(add_tensor_96, [0, 2, 3])
        sub_tensor_264 = torch.ops.aten.sub.Tensor(arg1472_1, arg1813_1);  arg1472_1 = arg1813_1 = None
        mul_tensor_744 = torch.ops.aten.mul.Tensor(add_tensor_96, sub_tensor_264)
        sum_dim_int_list_141 = torch.ops.aten.sum.dim_IntList(mul_tensor_744, [0, 2, 3]);  mul_tensor_744 = None
        mul_tensor_745 = torch.ops.aten.mul.Tensor(sum_dim_int_list_140, 0.008264462809917356);  sum_dim_int_list_140 = None
        view_default_229 = torch.ops.aten.view.default(mul_tensor_745, [1, 672, 1, 1]);  mul_tensor_745 = None
        mul_tensor_746 = torch.ops.aten.mul.Tensor(sum_dim_int_list_141, 0.008264462809917356)
        mul_tensor_747 = torch.ops.aten.mul.Tensor(arg1473_1, arg1473_1)
        mul_tensor_748 = torch.ops.aten.mul.Tensor(mul_tensor_746, mul_tensor_747);  mul_tensor_746 = mul_tensor_747 = None
        view_default_230 = torch.ops.aten.view.default(mul_tensor_748, [1, 672, 1, 1]);  mul_tensor_748 = None
        mul_tensor_749 = torch.ops.aten.mul.Tensor(arg1473_1, arg558_1);  arg558_1 = None
        view_default_231 = torch.ops.aten.view.default(mul_tensor_749, [1, 672, 1, 1]);  mul_tensor_749 = None
        mul_tensor_750 = torch.ops.aten.mul.Tensor(sub_tensor_264, view_default_230);  sub_tensor_264 = view_default_230 = None
        sub_tensor_265 = torch.ops.aten.sub.Tensor(add_tensor_96, mul_tensor_750);  mul_tensor_750 = None
        sub_tensor_266 = torch.ops.aten.sub.Tensor(sub_tensor_265, view_default_229);  sub_tensor_265 = view_default_229 = None
        mul_tensor_751 = torch.ops.aten.mul.Tensor(sub_tensor_266, view_default_231);  sub_tensor_266 = view_default_231 = None
        mul_tensor_752 = torch.ops.aten.mul.Tensor(sum_dim_int_list_141, arg1473_1);  sum_dim_int_list_141 = arg1473_1 = None
        convolution_backward_default_141 = torch.ops.aten.convolution_backward.default(mul_tensor_751, arg1471_1, arg557_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_751 = arg1471_1 = arg557_1 = None
        getitem_423 = convolution_backward_default_141[0]
        getitem_424 = convolution_backward_default_141[1];  convolution_backward_default_141 = None
        convolution_backward_default_142 = torch.ops.aten.convolution_backward.default(getitem_423, arg1470_1, arg556_1, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 672, [True, True, False]);  getitem_423 = arg556_1 = None
        getitem_426 = convolution_backward_default_142[0]
        getitem_427 = convolution_backward_default_142[1];  convolution_backward_default_142 = None
        le_scalar_52 = torch.ops.aten.le.Scalar(arg1470_1, 0);  arg1470_1 = None
        new_zeros_default_77 = torch.ops.aten.new_zeros.default(getitem_426, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_77 = torch.ops.aten.where.self(le_scalar_52, new_zeros_default_77, getitem_426);  le_scalar_52 = new_zeros_default_77 = getitem_426 = None
        sum_dim_int_list_142 = torch.ops.aten.sum.dim_IntList(where_self_77, [0, 2, 3])
        sub_tensor_267 = torch.ops.aten.sub.Tensor(arg1468_1, arg1814_1);  arg1468_1 = arg1814_1 = None
        mul_tensor_753 = torch.ops.aten.mul.Tensor(where_self_77, sub_tensor_267)
        sum_dim_int_list_143 = torch.ops.aten.sum.dim_IntList(mul_tensor_753, [0, 2, 3]);  mul_tensor_753 = None
        mul_tensor_754 = torch.ops.aten.mul.Tensor(sum_dim_int_list_142, 0.008264462809917356);  sum_dim_int_list_142 = None
        view_default_232 = torch.ops.aten.view.default(mul_tensor_754, [1, 672, 1, 1]);  mul_tensor_754 = None
        mul_tensor_755 = torch.ops.aten.mul.Tensor(sum_dim_int_list_143, 0.008264462809917356)
        mul_tensor_756 = torch.ops.aten.mul.Tensor(arg1469_1, arg1469_1)
        mul_tensor_757 = torch.ops.aten.mul.Tensor(mul_tensor_755, mul_tensor_756);  mul_tensor_755 = mul_tensor_756 = None
        view_default_233 = torch.ops.aten.view.default(mul_tensor_757, [1, 672, 1, 1]);  mul_tensor_757 = None
        mul_tensor_758 = torch.ops.aten.mul.Tensor(arg1469_1, arg555_1);  arg555_1 = None
        view_default_234 = torch.ops.aten.view.default(mul_tensor_758, [1, 672, 1, 1]);  mul_tensor_758 = None
        mul_tensor_759 = torch.ops.aten.mul.Tensor(sub_tensor_267, view_default_233);  sub_tensor_267 = view_default_233 = None
        sub_tensor_268 = torch.ops.aten.sub.Tensor(where_self_77, mul_tensor_759);  where_self_77 = mul_tensor_759 = None
        sub_tensor_269 = torch.ops.aten.sub.Tensor(sub_tensor_268, view_default_232);  sub_tensor_268 = view_default_232 = None
        mul_tensor_760 = torch.ops.aten.mul.Tensor(sub_tensor_269, view_default_234);  sub_tensor_269 = view_default_234 = None
        mul_tensor_761 = torch.ops.aten.mul.Tensor(sum_dim_int_list_143, arg1469_1);  sum_dim_int_list_143 = arg1469_1 = None
        convolution_backward_default_143 = torch.ops.aten.convolution_backward.default(mul_tensor_760, arg1467_1, arg554_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_760 = arg1467_1 = arg554_1 = None
        getitem_429 = convolution_backward_default_143[0]
        getitem_430 = convolution_backward_default_143[1];  convolution_backward_default_143 = None
        convolution_backward_default_144 = torch.ops.aten.convolution_backward.default(getitem_429, arg1456_1, arg14_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 672, [True, True, False]);  getitem_429 = arg14_1 = None
        getitem_432 = convolution_backward_default_144[0]
        getitem_433 = convolution_backward_default_144[1];  convolution_backward_default_144 = None
        constant_pad_nd_default_4 = torch.ops.aten.constant_pad_nd.default(getitem_432, [-3, -3, -3, -3]);  getitem_432 = None
        new_zeros_default_78 = torch.ops.aten.new_zeros.default(constant_pad_nd_default_4, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_78 = torch.ops.aten.where.self(arg1812_1, new_zeros_default_78, constant_pad_nd_default_4);  new_zeros_default_78 = constant_pad_nd_default_4 = None
        add_tensor_99 = torch.ops.aten.add.Tensor(where_self_76, where_self_78);  where_self_76 = where_self_78 = None
        max_pool2d_with_indices_backward_default_1 = torch.ops.aten.max_pool2d_with_indices_backward.default(add_tensor_96, arg1465_1, [3, 3], [2, 2], [0, 0], [1, 1], False, arg1466_1);  add_tensor_96 = arg1465_1 = arg1466_1 = None
        constant_pad_nd_default_5 = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_default_1, [-1, -1, -1, -1]);  max_pool2d_with_indices_backward_default_1 = None
        add_tensor_100 = torch.ops.aten.add.Tensor(add_tensor_98, constant_pad_nd_default_5);  add_tensor_98 = constant_pad_nd_default_5 = None
        sum_dim_int_list_144 = torch.ops.aten.sum.dim_IntList(add_tensor_97, [0, 2, 3])
        sub_tensor_270 = torch.ops.aten.sub.Tensor(arg1462_1, arg1815_1);  arg1462_1 = arg1815_1 = None
        mul_tensor_762 = torch.ops.aten.mul.Tensor(add_tensor_97, sub_tensor_270)
        sum_dim_int_list_145 = torch.ops.aten.sum.dim_IntList(mul_tensor_762, [0, 2, 3]);  mul_tensor_762 = None
        mul_tensor_763 = torch.ops.aten.mul.Tensor(sum_dim_int_list_144, 0.008264462809917356);  sum_dim_int_list_144 = None
        view_default_235 = torch.ops.aten.view.default(mul_tensor_763, [1, 672, 1, 1]);  mul_tensor_763 = None
        mul_tensor_764 = torch.ops.aten.mul.Tensor(sum_dim_int_list_145, 0.008264462809917356)
        mul_tensor_765 = torch.ops.aten.mul.Tensor(arg1463_1, arg1463_1)
        mul_tensor_766 = torch.ops.aten.mul.Tensor(mul_tensor_764, mul_tensor_765);  mul_tensor_764 = mul_tensor_765 = None
        view_default_236 = torch.ops.aten.view.default(mul_tensor_766, [1, 672, 1, 1]);  mul_tensor_766 = None
        mul_tensor_767 = torch.ops.aten.mul.Tensor(arg1463_1, arg553_1);  arg553_1 = None
        view_default_237 = torch.ops.aten.view.default(mul_tensor_767, [1, 672, 1, 1]);  mul_tensor_767 = None
        mul_tensor_768 = torch.ops.aten.mul.Tensor(sub_tensor_270, view_default_236);  sub_tensor_270 = view_default_236 = None
        sub_tensor_271 = torch.ops.aten.sub.Tensor(add_tensor_97, mul_tensor_768);  mul_tensor_768 = None
        sub_tensor_272 = torch.ops.aten.sub.Tensor(sub_tensor_271, view_default_235);  sub_tensor_271 = None
        mul_tensor_769 = torch.ops.aten.mul.Tensor(sub_tensor_272, view_default_237);  sub_tensor_272 = view_default_237 = None
        mul_tensor_770 = torch.ops.aten.mul.Tensor(sum_dim_int_list_145, arg1463_1);  sum_dim_int_list_145 = arg1463_1 = None
        convolution_backward_default_145 = torch.ops.aten.convolution_backward.default(mul_tensor_769, arg1461_1, arg552_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_769 = arg1461_1 = arg552_1 = None
        getitem_435 = convolution_backward_default_145[0]
        getitem_436 = convolution_backward_default_145[1];  convolution_backward_default_145 = None
        convolution_backward_default_146 = torch.ops.aten.convolution_backward.default(getitem_435, arg1460_1, arg551_1, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 672, [True, True, False]);  getitem_435 = arg551_1 = None
        getitem_438 = convolution_backward_default_146[0]
        getitem_439 = convolution_backward_default_146[1];  convolution_backward_default_146 = None
        le_scalar_53 = torch.ops.aten.le.Scalar(arg1460_1, 0);  arg1460_1 = None
        new_zeros_default_79 = torch.ops.aten.new_zeros.default(getitem_438, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_79 = torch.ops.aten.where.self(le_scalar_53, new_zeros_default_79, getitem_438);  le_scalar_53 = new_zeros_default_79 = getitem_438 = None
        sum_dim_int_list_146 = torch.ops.aten.sum.dim_IntList(where_self_79, [0, 2, 3])
        sub_tensor_273 = torch.ops.aten.sub.Tensor(arg1458_1, arg1816_1);  arg1458_1 = arg1816_1 = None
        mul_tensor_771 = torch.ops.aten.mul.Tensor(where_self_79, sub_tensor_273)
        sum_dim_int_list_147 = torch.ops.aten.sum.dim_IntList(mul_tensor_771, [0, 2, 3]);  mul_tensor_771 = None
        mul_tensor_772 = torch.ops.aten.mul.Tensor(sum_dim_int_list_146, 0.008264462809917356);  sum_dim_int_list_146 = None
        view_default_238 = torch.ops.aten.view.default(mul_tensor_772, [1, 672, 1, 1]);  mul_tensor_772 = None
        mul_tensor_773 = torch.ops.aten.mul.Tensor(sum_dim_int_list_147, 0.008264462809917356)
        mul_tensor_774 = torch.ops.aten.mul.Tensor(arg1459_1, arg1459_1)
        mul_tensor_775 = torch.ops.aten.mul.Tensor(mul_tensor_773, mul_tensor_774);  mul_tensor_773 = mul_tensor_774 = None
        view_default_239 = torch.ops.aten.view.default(mul_tensor_775, [1, 672, 1, 1]);  mul_tensor_775 = None
        mul_tensor_776 = torch.ops.aten.mul.Tensor(arg1459_1, arg550_1);  arg550_1 = None
        view_default_240 = torch.ops.aten.view.default(mul_tensor_776, [1, 672, 1, 1]);  mul_tensor_776 = None
        mul_tensor_777 = torch.ops.aten.mul.Tensor(sub_tensor_273, view_default_239);  sub_tensor_273 = view_default_239 = None
        sub_tensor_274 = torch.ops.aten.sub.Tensor(where_self_79, mul_tensor_777);  where_self_79 = mul_tensor_777 = None
        sub_tensor_275 = torch.ops.aten.sub.Tensor(sub_tensor_274, view_default_238);  sub_tensor_274 = view_default_238 = None
        mul_tensor_778 = torch.ops.aten.mul.Tensor(sub_tensor_275, view_default_240);  sub_tensor_275 = view_default_240 = None
        mul_tensor_779 = torch.ops.aten.mul.Tensor(sum_dim_int_list_147, arg1459_1);  sum_dim_int_list_147 = arg1459_1 = None
        convolution_backward_default_147 = torch.ops.aten.convolution_backward.default(mul_tensor_778, arg1457_1, arg549_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_778 = arg1457_1 = arg549_1 = None
        getitem_441 = convolution_backward_default_147[0]
        getitem_442 = convolution_backward_default_147[1];  convolution_backward_default_147 = None
        convolution_backward_default_148 = torch.ops.aten.convolution_backward.default(getitem_441, arg1456_1, arg13_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 672, [True, True, False]);  getitem_441 = arg1456_1 = arg13_1 = None
        getitem_444 = convolution_backward_default_148[0]
        getitem_445 = convolution_backward_default_148[1];  convolution_backward_default_148 = None
        constant_pad_nd_default_6 = torch.ops.aten.constant_pad_nd.default(getitem_444, [-3, -3, -3, -3]);  getitem_444 = None
        new_zeros_default_80 = torch.ops.aten.new_zeros.default(constant_pad_nd_default_6, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_80 = torch.ops.aten.where.self(arg1812_1, new_zeros_default_80, constant_pad_nd_default_6);  arg1812_1 = new_zeros_default_80 = constant_pad_nd_default_6 = None
        add_tensor_101 = torch.ops.aten.add.Tensor(add_tensor_99, where_self_80);  add_tensor_99 = where_self_80 = None
        sub_tensor_276 = torch.ops.aten.sub.Tensor(arg1454_1, arg1817_1);  arg1454_1 = arg1817_1 = None
        mul_tensor_780 = torch.ops.aten.mul.Tensor(add_tensor_97, sub_tensor_276)
        sum_dim_int_list_148 = torch.ops.aten.sum.dim_IntList(mul_tensor_780, [0, 2, 3]);  mul_tensor_780 = None
        mul_tensor_781 = torch.ops.aten.mul.Tensor(sum_dim_int_list_148, 0.008264462809917356)
        mul_tensor_782 = torch.ops.aten.mul.Tensor(arg1455_1, arg1455_1)
        mul_tensor_783 = torch.ops.aten.mul.Tensor(mul_tensor_781, mul_tensor_782);  mul_tensor_781 = mul_tensor_782 = None
        view_default_241 = torch.ops.aten.view.default(mul_tensor_783, [1, 672, 1, 1]);  mul_tensor_783 = None
        mul_tensor_784 = torch.ops.aten.mul.Tensor(arg1455_1, arg548_1);  arg548_1 = None
        view_default_242 = torch.ops.aten.view.default(mul_tensor_784, [1, 672, 1, 1]);  mul_tensor_784 = None
        mul_tensor_785 = torch.ops.aten.mul.Tensor(sub_tensor_276, view_default_241);  sub_tensor_276 = view_default_241 = None
        sub_tensor_277 = torch.ops.aten.sub.Tensor(add_tensor_97, mul_tensor_785);  add_tensor_97 = mul_tensor_785 = None
        sub_tensor_278 = torch.ops.aten.sub.Tensor(sub_tensor_277, view_default_235);  sub_tensor_277 = view_default_235 = None
        mul_tensor_786 = torch.ops.aten.mul.Tensor(sub_tensor_278, view_default_242);  sub_tensor_278 = view_default_242 = None
        mul_tensor_787 = torch.ops.aten.mul.Tensor(sum_dim_int_list_148, arg1455_1);  sum_dim_int_list_148 = arg1455_1 = None
        convolution_backward_default_149 = torch.ops.aten.convolution_backward.default(mul_tensor_786, arg1453_1, arg547_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_786 = arg1453_1 = arg547_1 = None
        getitem_447 = convolution_backward_default_149[0]
        getitem_448 = convolution_backward_default_149[1];  convolution_backward_default_149 = None
        convolution_backward_default_150 = torch.ops.aten.convolution_backward.default(getitem_447, arg1452_1, arg546_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  getitem_447 = arg546_1 = None
        getitem_450 = convolution_backward_default_150[0]
        getitem_451 = convolution_backward_default_150[1];  convolution_backward_default_150 = None
        le_scalar_54 = torch.ops.aten.le.Scalar(arg1452_1, 0);  arg1452_1 = None
        new_zeros_default_81 = torch.ops.aten.new_zeros.default(getitem_450, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_81 = torch.ops.aten.where.self(le_scalar_54, new_zeros_default_81, getitem_450);  le_scalar_54 = new_zeros_default_81 = getitem_450 = None
        sum_dim_int_list_149 = torch.ops.aten.sum.dim_IntList(where_self_81, [0, 2, 3])
        sub_tensor_279 = torch.ops.aten.sub.Tensor(arg1450_1, arg1818_1);  arg1450_1 = arg1818_1 = None
        mul_tensor_788 = torch.ops.aten.mul.Tensor(where_self_81, sub_tensor_279)
        sum_dim_int_list_150 = torch.ops.aten.sum.dim_IntList(mul_tensor_788, [0, 2, 3]);  mul_tensor_788 = None
        mul_tensor_789 = torch.ops.aten.mul.Tensor(sum_dim_int_list_149, 0.008264462809917356);  sum_dim_int_list_149 = None
        view_default_243 = torch.ops.aten.view.default(mul_tensor_789, [1, 672, 1, 1]);  mul_tensor_789 = None
        mul_tensor_790 = torch.ops.aten.mul.Tensor(sum_dim_int_list_150, 0.008264462809917356)
        mul_tensor_791 = torch.ops.aten.mul.Tensor(arg1451_1, arg1451_1)
        mul_tensor_792 = torch.ops.aten.mul.Tensor(mul_tensor_790, mul_tensor_791);  mul_tensor_790 = mul_tensor_791 = None
        view_default_244 = torch.ops.aten.view.default(mul_tensor_792, [1, 672, 1, 1]);  mul_tensor_792 = None
        mul_tensor_793 = torch.ops.aten.mul.Tensor(arg1451_1, arg545_1);  arg545_1 = None
        view_default_245 = torch.ops.aten.view.default(mul_tensor_793, [1, 672, 1, 1]);  mul_tensor_793 = None
        mul_tensor_794 = torch.ops.aten.mul.Tensor(sub_tensor_279, view_default_244);  sub_tensor_279 = view_default_244 = None
        sub_tensor_280 = torch.ops.aten.sub.Tensor(where_self_81, mul_tensor_794);  where_self_81 = mul_tensor_794 = None
        sub_tensor_281 = torch.ops.aten.sub.Tensor(sub_tensor_280, view_default_243);  sub_tensor_280 = view_default_243 = None
        mul_tensor_795 = torch.ops.aten.mul.Tensor(sub_tensor_281, view_default_245);  sub_tensor_281 = view_default_245 = None
        mul_tensor_796 = torch.ops.aten.mul.Tensor(sum_dim_int_list_150, arg1451_1);  sum_dim_int_list_150 = arg1451_1 = None
        convolution_backward_default_151 = torch.ops.aten.convolution_backward.default(mul_tensor_795, arg1449_1, arg544_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_795 = arg1449_1 = arg544_1 = None
        getitem_453 = convolution_backward_default_151[0]
        getitem_454 = convolution_backward_default_151[1];  convolution_backward_default_151 = None
        convolution_backward_default_152 = torch.ops.aten.convolution_backward.default(getitem_453, arg1448_1, arg12_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 672, [True, True, False]);  getitem_453 = arg1448_1 = arg12_1 = None
        getitem_456 = convolution_backward_default_152[0]
        getitem_457 = convolution_backward_default_152[1];  convolution_backward_default_152 = None
        constant_pad_nd_default_7 = torch.ops.aten.constant_pad_nd.default(getitem_456, [-2, -2, -2, -2]);  getitem_456 = None
        new_zeros_default_82 = torch.ops.aten.new_zeros.default(constant_pad_nd_default_7, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_82 = torch.ops.aten.where.self(arg1819_1, new_zeros_default_82, constant_pad_nd_default_7);  arg1819_1 = new_zeros_default_82 = constant_pad_nd_default_7 = None
        add_tensor_102 = torch.ops.aten.add.Tensor(add_tensor_100, where_self_82);  add_tensor_100 = where_self_82 = None
        sum_dim_int_list_151 = torch.ops.aten.sum.dim_IntList(add_tensor_102, [0, 2, 3])
        sub_tensor_282 = torch.ops.aten.sub.Tensor(arg1446_1, arg1820_1);  arg1446_1 = arg1820_1 = None
        mul_tensor_797 = torch.ops.aten.mul.Tensor(add_tensor_102, sub_tensor_282)
        sum_dim_int_list_152 = torch.ops.aten.sum.dim_IntList(mul_tensor_797, [0, 2, 3]);  mul_tensor_797 = None
        mul_tensor_798 = torch.ops.aten.mul.Tensor(sum_dim_int_list_151, 0.0022675736961451248);  sum_dim_int_list_151 = None
        view_default_246 = torch.ops.aten.view.default(mul_tensor_798, [1, 672, 1, 1]);  mul_tensor_798 = None
        mul_tensor_799 = torch.ops.aten.mul.Tensor(sum_dim_int_list_152, 0.0022675736961451248)
        mul_tensor_800 = torch.ops.aten.mul.Tensor(arg1447_1, arg1447_1)
        mul_tensor_801 = torch.ops.aten.mul.Tensor(mul_tensor_799, mul_tensor_800);  mul_tensor_799 = mul_tensor_800 = None
        view_default_247 = torch.ops.aten.view.default(mul_tensor_801, [1, 672, 1, 1]);  mul_tensor_801 = None
        mul_tensor_802 = torch.ops.aten.mul.Tensor(arg1447_1, arg543_1);  arg543_1 = None
        view_default_248 = torch.ops.aten.view.default(mul_tensor_802, [1, 672, 1, 1]);  mul_tensor_802 = None
        mul_tensor_803 = torch.ops.aten.mul.Tensor(sub_tensor_282, view_default_247);  sub_tensor_282 = view_default_247 = None
        sub_tensor_283 = torch.ops.aten.sub.Tensor(add_tensor_102, mul_tensor_803);  add_tensor_102 = mul_tensor_803 = None
        sub_tensor_284 = torch.ops.aten.sub.Tensor(sub_tensor_283, view_default_246);  sub_tensor_283 = view_default_246 = None
        mul_tensor_804 = torch.ops.aten.mul.Tensor(sub_tensor_284, view_default_248);  sub_tensor_284 = view_default_248 = None
        mul_tensor_805 = torch.ops.aten.mul.Tensor(sum_dim_int_list_152, arg1447_1);  sum_dim_int_list_152 = arg1447_1 = None
        convolution_backward_default_153 = torch.ops.aten.convolution_backward.default(mul_tensor_804, arg1445_1, arg542_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_804 = arg542_1 = None
        getitem_459 = convolution_backward_default_153[0]
        getitem_460 = convolution_backward_default_153[1];  convolution_backward_default_153 = None
        le_scalar_55 = torch.ops.aten.le.Scalar(arg1445_1, 0);  arg1445_1 = None
        new_zeros_default_83 = torch.ops.aten.new_zeros.default(getitem_459, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_83 = torch.ops.aten.where.self(le_scalar_55, new_zeros_default_83, getitem_459);  le_scalar_55 = new_zeros_default_83 = getitem_459 = None
        sum_dim_int_list_153 = torch.ops.aten.sum.dim_IntList(add_tensor_101, [0, 2, 3])
        sub_tensor_285 = torch.ops.aten.sub.Tensor(arg1443_1, arg1821_1);  arg1443_1 = arg1821_1 = None
        mul_tensor_806 = torch.ops.aten.mul.Tensor(add_tensor_101, sub_tensor_285)
        sum_dim_int_list_154 = torch.ops.aten.sum.dim_IntList(mul_tensor_806, [0, 2, 3]);  mul_tensor_806 = None
        mul_tensor_807 = torch.ops.aten.mul.Tensor(sum_dim_int_list_153, 0.0022675736961451248);  sum_dim_int_list_153 = None
        view_default_249 = torch.ops.aten.view.default(mul_tensor_807, [1, 672, 1, 1]);  mul_tensor_807 = None
        mul_tensor_808 = torch.ops.aten.mul.Tensor(sum_dim_int_list_154, 0.0022675736961451248)
        mul_tensor_809 = torch.ops.aten.mul.Tensor(arg1444_1, arg1444_1)
        mul_tensor_810 = torch.ops.aten.mul.Tensor(mul_tensor_808, mul_tensor_809);  mul_tensor_808 = mul_tensor_809 = None
        view_default_250 = torch.ops.aten.view.default(mul_tensor_810, [1, 672, 1, 1]);  mul_tensor_810 = None
        mul_tensor_811 = torch.ops.aten.mul.Tensor(arg1444_1, arg541_1);  arg541_1 = None
        view_default_251 = torch.ops.aten.view.default(mul_tensor_811, [1, 672, 1, 1]);  mul_tensor_811 = None
        mul_tensor_812 = torch.ops.aten.mul.Tensor(sub_tensor_285, view_default_250);  sub_tensor_285 = view_default_250 = None
        sub_tensor_286 = torch.ops.aten.sub.Tensor(add_tensor_101, mul_tensor_812);  add_tensor_101 = mul_tensor_812 = None
        sub_tensor_287 = torch.ops.aten.sub.Tensor(sub_tensor_286, view_default_249);  sub_tensor_286 = view_default_249 = None
        mul_tensor_813 = torch.ops.aten.mul.Tensor(sub_tensor_287, view_default_251);  sub_tensor_287 = view_default_251 = None
        mul_tensor_814 = torch.ops.aten.mul.Tensor(sum_dim_int_list_154, arg1444_1);  sum_dim_int_list_154 = arg1444_1 = None
        convolution_backward_default_154 = torch.ops.aten.convolution_backward.default(mul_tensor_813, arg1404_1, arg540_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_813 = arg540_1 = None
        getitem_462 = convolution_backward_default_154[0]
        getitem_463 = convolution_backward_default_154[1];  convolution_backward_default_154 = None
        new_zeros_default_84 = torch.ops.aten.new_zeros.default(getitem_462, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_84 = torch.ops.aten.where.self(le_scalar_48, new_zeros_default_84, getitem_462);  new_zeros_default_84 = getitem_462 = None
        add_tensor_103 = torch.ops.aten.add.Tensor(where_self_72, where_self_84);  where_self_72 = where_self_84 = None
        slice_tensor_42 = torch.ops.aten.slice.Tensor(where_self_83, 1, 0, 336)
        slice_tensor_43 = torch.ops.aten.slice.Tensor(where_self_83, 1, 336, 672)
        slice_tensor_44 = torch.ops.aten.slice.Tensor(where_self_83, 1, 672, 1008)
        slice_tensor_45 = torch.ops.aten.slice.Tensor(where_self_83, 1, 1008, 1344)
        slice_tensor_46 = torch.ops.aten.slice.Tensor(where_self_83, 1, 1344, 1680)
        slice_tensor_47 = torch.ops.aten.slice.Tensor(where_self_83, 1, 1680, 2016);  where_self_83 = None
        sum_dim_int_list_155 = torch.ops.aten.sum.dim_IntList(slice_tensor_47, [0, 2, 3])
        sub_tensor_288 = torch.ops.aten.sub.Tensor(arg1441_1, arg1822_1);  arg1441_1 = arg1822_1 = None
        mul_tensor_815 = torch.ops.aten.mul.Tensor(slice_tensor_47, sub_tensor_288)
        sum_dim_int_list_156 = torch.ops.aten.sum.dim_IntList(mul_tensor_815, [0, 2, 3]);  mul_tensor_815 = None
        mul_tensor_816 = torch.ops.aten.mul.Tensor(sum_dim_int_list_155, 0.0022675736961451248);  sum_dim_int_list_155 = None
        view_default_252 = torch.ops.aten.view.default(mul_tensor_816, [1, 336, 1, 1]);  mul_tensor_816 = None
        mul_tensor_817 = torch.ops.aten.mul.Tensor(sum_dim_int_list_156, 0.0022675736961451248)
        mul_tensor_818 = torch.ops.aten.mul.Tensor(arg1442_1, arg1442_1)
        mul_tensor_819 = torch.ops.aten.mul.Tensor(mul_tensor_817, mul_tensor_818);  mul_tensor_817 = mul_tensor_818 = None
        view_default_253 = torch.ops.aten.view.default(mul_tensor_819, [1, 336, 1, 1]);  mul_tensor_819 = None
        mul_tensor_820 = torch.ops.aten.mul.Tensor(arg1442_1, arg539_1);  arg539_1 = None
        view_default_254 = torch.ops.aten.view.default(mul_tensor_820, [1, 336, 1, 1]);  mul_tensor_820 = None
        mul_tensor_821 = torch.ops.aten.mul.Tensor(sub_tensor_288, view_default_253);  sub_tensor_288 = view_default_253 = None
        sub_tensor_289 = torch.ops.aten.sub.Tensor(slice_tensor_47, mul_tensor_821);  mul_tensor_821 = None
        sub_tensor_290 = torch.ops.aten.sub.Tensor(sub_tensor_289, view_default_252);  sub_tensor_289 = view_default_252 = None
        mul_tensor_822 = torch.ops.aten.mul.Tensor(sub_tensor_290, view_default_254);  sub_tensor_290 = view_default_254 = None
        mul_tensor_823 = torch.ops.aten.mul.Tensor(sum_dim_int_list_156, arg1442_1);  sum_dim_int_list_156 = arg1442_1 = None
        convolution_backward_default_155 = torch.ops.aten.convolution_backward.default(mul_tensor_822, arg1440_1, arg538_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_822 = arg1440_1 = arg538_1 = None
        getitem_465 = convolution_backward_default_155[0]
        getitem_466 = convolution_backward_default_155[1];  convolution_backward_default_155 = None
        convolution_backward_default_156 = torch.ops.aten.convolution_backward.default(getitem_465, arg1439_1, arg537_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_465 = arg537_1 = None
        getitem_468 = convolution_backward_default_156[0]
        getitem_469 = convolution_backward_default_156[1];  convolution_backward_default_156 = None
        le_scalar_56 = torch.ops.aten.le.Scalar(arg1439_1, 0);  arg1439_1 = None
        new_zeros_default_85 = torch.ops.aten.new_zeros.default(getitem_468, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_85 = torch.ops.aten.where.self(le_scalar_56, new_zeros_default_85, getitem_468);  le_scalar_56 = new_zeros_default_85 = getitem_468 = None
        sum_dim_int_list_157 = torch.ops.aten.sum.dim_IntList(where_self_85, [0, 2, 3])
        sub_tensor_291 = torch.ops.aten.sub.Tensor(arg1437_1, arg1823_1);  arg1437_1 = arg1823_1 = None
        mul_tensor_824 = torch.ops.aten.mul.Tensor(where_self_85, sub_tensor_291)
        sum_dim_int_list_158 = torch.ops.aten.sum.dim_IntList(mul_tensor_824, [0, 2, 3]);  mul_tensor_824 = None
        mul_tensor_825 = torch.ops.aten.mul.Tensor(sum_dim_int_list_157, 0.0022675736961451248);  sum_dim_int_list_157 = None
        view_default_255 = torch.ops.aten.view.default(mul_tensor_825, [1, 336, 1, 1]);  mul_tensor_825 = None
        mul_tensor_826 = torch.ops.aten.mul.Tensor(sum_dim_int_list_158, 0.0022675736961451248)
        mul_tensor_827 = torch.ops.aten.mul.Tensor(arg1438_1, arg1438_1)
        mul_tensor_828 = torch.ops.aten.mul.Tensor(mul_tensor_826, mul_tensor_827);  mul_tensor_826 = mul_tensor_827 = None
        view_default_256 = torch.ops.aten.view.default(mul_tensor_828, [1, 336, 1, 1]);  mul_tensor_828 = None
        mul_tensor_829 = torch.ops.aten.mul.Tensor(arg1438_1, arg536_1);  arg536_1 = None
        view_default_257 = torch.ops.aten.view.default(mul_tensor_829, [1, 336, 1, 1]);  mul_tensor_829 = None
        mul_tensor_830 = torch.ops.aten.mul.Tensor(sub_tensor_291, view_default_256);  sub_tensor_291 = view_default_256 = None
        sub_tensor_292 = torch.ops.aten.sub.Tensor(where_self_85, mul_tensor_830);  where_self_85 = mul_tensor_830 = None
        sub_tensor_293 = torch.ops.aten.sub.Tensor(sub_tensor_292, view_default_255);  sub_tensor_292 = view_default_255 = None
        mul_tensor_831 = torch.ops.aten.mul.Tensor(sub_tensor_293, view_default_257);  sub_tensor_293 = view_default_257 = None
        mul_tensor_832 = torch.ops.aten.mul.Tensor(sum_dim_int_list_158, arg1438_1);  sum_dim_int_list_158 = arg1438_1 = None
        convolution_backward_default_157 = torch.ops.aten.convolution_backward.default(mul_tensor_831, arg1436_1, arg535_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_831 = arg1436_1 = arg535_1 = None
        getitem_471 = convolution_backward_default_157[0]
        getitem_472 = convolution_backward_default_157[1];  convolution_backward_default_157 = None
        convolution_backward_default_158 = torch.ops.aten.convolution_backward.default(getitem_471, relu_default_25, arg534_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_471 = arg534_1 = None
        getitem_474 = convolution_backward_default_158[0]
        getitem_475 = convolution_backward_default_158[1];  convolution_backward_default_158 = None
        le_scalar_57 = torch.ops.aten.le.Scalar(relu_default_25, 0)
        new_zeros_default_86 = torch.ops.aten.new_zeros.default(getitem_474, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_86 = torch.ops.aten.where.self(le_scalar_57, new_zeros_default_86, getitem_474);  new_zeros_default_86 = getitem_474 = None
        add_tensor_104 = torch.ops.aten.add.Tensor(slice_tensor_47, where_self_86);  slice_tensor_47 = where_self_86 = None
        avg_pool2d_backward_default_16 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_46, add_tensor_22, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_46 = add_tensor_22 = None
        add_tensor_105 = torch.ops.aten.add.Tensor(slice_tensor_42, avg_pool2d_backward_default_16);  slice_tensor_42 = None
        add_tensor_106 = torch.ops.aten.add.Tensor(add_tensor_105, avg_pool2d_backward_default_16);  add_tensor_105 = avg_pool2d_backward_default_16 = None
        add_tensor_107 = torch.ops.aten.add.Tensor(add_tensor_106, slice_tensor_45);  add_tensor_106 = None
        avg_pool2d_backward_default_17 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_45, add_tensor_23, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_45 = add_tensor_23 = None
        add_tensor_108 = torch.ops.aten.add.Tensor(add_tensor_104, avg_pool2d_backward_default_17);  add_tensor_104 = avg_pool2d_backward_default_17 = None
        sum_dim_int_list_159 = torch.ops.aten.sum.dim_IntList(slice_tensor_44, [0, 2, 3])
        sub_tensor_294 = torch.ops.aten.sub.Tensor(arg1434_1, arg1824_1);  arg1434_1 = arg1824_1 = None
        mul_tensor_833 = torch.ops.aten.mul.Tensor(slice_tensor_44, sub_tensor_294)
        sum_dim_int_list_160 = torch.ops.aten.sum.dim_IntList(mul_tensor_833, [0, 2, 3]);  mul_tensor_833 = None
        mul_tensor_834 = torch.ops.aten.mul.Tensor(sum_dim_int_list_159, 0.0022675736961451248);  sum_dim_int_list_159 = None
        view_default_258 = torch.ops.aten.view.default(mul_tensor_834, [1, 336, 1, 1]);  mul_tensor_834 = None
        mul_tensor_835 = torch.ops.aten.mul.Tensor(sum_dim_int_list_160, 0.0022675736961451248)
        mul_tensor_836 = torch.ops.aten.mul.Tensor(arg1435_1, arg1435_1)
        mul_tensor_837 = torch.ops.aten.mul.Tensor(mul_tensor_835, mul_tensor_836);  mul_tensor_835 = mul_tensor_836 = None
        view_default_259 = torch.ops.aten.view.default(mul_tensor_837, [1, 336, 1, 1]);  mul_tensor_837 = None
        mul_tensor_838 = torch.ops.aten.mul.Tensor(arg1435_1, arg533_1);  arg533_1 = None
        view_default_260 = torch.ops.aten.view.default(mul_tensor_838, [1, 336, 1, 1]);  mul_tensor_838 = None
        mul_tensor_839 = torch.ops.aten.mul.Tensor(sub_tensor_294, view_default_259);  sub_tensor_294 = view_default_259 = None
        sub_tensor_295 = torch.ops.aten.sub.Tensor(slice_tensor_44, mul_tensor_839);  mul_tensor_839 = None
        sub_tensor_296 = torch.ops.aten.sub.Tensor(sub_tensor_295, view_default_258);  sub_tensor_295 = None
        mul_tensor_840 = torch.ops.aten.mul.Tensor(sub_tensor_296, view_default_260);  sub_tensor_296 = view_default_260 = None
        mul_tensor_841 = torch.ops.aten.mul.Tensor(sum_dim_int_list_160, arg1435_1);  sum_dim_int_list_160 = arg1435_1 = None
        convolution_backward_default_159 = torch.ops.aten.convolution_backward.default(mul_tensor_840, arg1433_1, arg532_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_840 = arg1433_1 = arg532_1 = None
        getitem_477 = convolution_backward_default_159[0]
        getitem_478 = convolution_backward_default_159[1];  convolution_backward_default_159 = None
        convolution_backward_default_160 = torch.ops.aten.convolution_backward.default(getitem_477, arg1432_1, arg531_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_477 = arg531_1 = None
        getitem_480 = convolution_backward_default_160[0]
        getitem_481 = convolution_backward_default_160[1];  convolution_backward_default_160 = None
        le_scalar_58 = torch.ops.aten.le.Scalar(arg1432_1, 0);  arg1432_1 = None
        new_zeros_default_87 = torch.ops.aten.new_zeros.default(getitem_480, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_87 = torch.ops.aten.where.self(le_scalar_58, new_zeros_default_87, getitem_480);  le_scalar_58 = new_zeros_default_87 = getitem_480 = None
        sum_dim_int_list_161 = torch.ops.aten.sum.dim_IntList(where_self_87, [0, 2, 3])
        sub_tensor_297 = torch.ops.aten.sub.Tensor(arg1430_1, arg1825_1);  arg1430_1 = arg1825_1 = None
        mul_tensor_842 = torch.ops.aten.mul.Tensor(where_self_87, sub_tensor_297)
        sum_dim_int_list_162 = torch.ops.aten.sum.dim_IntList(mul_tensor_842, [0, 2, 3]);  mul_tensor_842 = None
        mul_tensor_843 = torch.ops.aten.mul.Tensor(sum_dim_int_list_161, 0.0022675736961451248);  sum_dim_int_list_161 = None
        view_default_261 = torch.ops.aten.view.default(mul_tensor_843, [1, 336, 1, 1]);  mul_tensor_843 = None
        mul_tensor_844 = torch.ops.aten.mul.Tensor(sum_dim_int_list_162, 0.0022675736961451248)
        mul_tensor_845 = torch.ops.aten.mul.Tensor(arg1431_1, arg1431_1)
        mul_tensor_846 = torch.ops.aten.mul.Tensor(mul_tensor_844, mul_tensor_845);  mul_tensor_844 = mul_tensor_845 = None
        view_default_262 = torch.ops.aten.view.default(mul_tensor_846, [1, 336, 1, 1]);  mul_tensor_846 = None
        mul_tensor_847 = torch.ops.aten.mul.Tensor(arg1431_1, arg530_1);  arg530_1 = None
        view_default_263 = torch.ops.aten.view.default(mul_tensor_847, [1, 336, 1, 1]);  mul_tensor_847 = None
        mul_tensor_848 = torch.ops.aten.mul.Tensor(sub_tensor_297, view_default_262);  sub_tensor_297 = view_default_262 = None
        sub_tensor_298 = torch.ops.aten.sub.Tensor(where_self_87, mul_tensor_848);  where_self_87 = mul_tensor_848 = None
        sub_tensor_299 = torch.ops.aten.sub.Tensor(sub_tensor_298, view_default_261);  sub_tensor_298 = view_default_261 = None
        mul_tensor_849 = torch.ops.aten.mul.Tensor(sub_tensor_299, view_default_263);  sub_tensor_299 = view_default_263 = None
        mul_tensor_850 = torch.ops.aten.mul.Tensor(sum_dim_int_list_162, arg1431_1);  sum_dim_int_list_162 = arg1431_1 = None
        convolution_backward_default_161 = torch.ops.aten.convolution_backward.default(mul_tensor_849, arg1429_1, arg529_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_849 = arg1429_1 = arg529_1 = None
        getitem_483 = convolution_backward_default_161[0]
        getitem_484 = convolution_backward_default_161[1];  convolution_backward_default_161 = None
        convolution_backward_default_162 = torch.ops.aten.convolution_backward.default(getitem_483, relu_default_26, arg528_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_483 = arg528_1 = None
        getitem_486 = convolution_backward_default_162[0]
        getitem_487 = convolution_backward_default_162[1];  convolution_backward_default_162 = None
        le_scalar_59 = torch.ops.aten.le.Scalar(relu_default_26, 0)
        new_zeros_default_88 = torch.ops.aten.new_zeros.default(getitem_486, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_88 = torch.ops.aten.where.self(le_scalar_59, new_zeros_default_88, getitem_486);  new_zeros_default_88 = getitem_486 = None
        add_tensor_109 = torch.ops.aten.add.Tensor(add_tensor_107, where_self_88);  add_tensor_107 = where_self_88 = None
        sub_tensor_300 = torch.ops.aten.sub.Tensor(arg1427_1, arg1826_1);  arg1427_1 = arg1826_1 = None
        mul_tensor_851 = torch.ops.aten.mul.Tensor(slice_tensor_44, sub_tensor_300)
        sum_dim_int_list_163 = torch.ops.aten.sum.dim_IntList(mul_tensor_851, [0, 2, 3]);  mul_tensor_851 = None
        mul_tensor_852 = torch.ops.aten.mul.Tensor(sum_dim_int_list_163, 0.0022675736961451248)
        mul_tensor_853 = torch.ops.aten.mul.Tensor(arg1428_1, arg1428_1)
        mul_tensor_854 = torch.ops.aten.mul.Tensor(mul_tensor_852, mul_tensor_853);  mul_tensor_852 = mul_tensor_853 = None
        view_default_264 = torch.ops.aten.view.default(mul_tensor_854, [1, 336, 1, 1]);  mul_tensor_854 = None
        mul_tensor_855 = torch.ops.aten.mul.Tensor(arg1428_1, arg527_1);  arg527_1 = None
        view_default_265 = torch.ops.aten.view.default(mul_tensor_855, [1, 336, 1, 1]);  mul_tensor_855 = None
        mul_tensor_856 = torch.ops.aten.mul.Tensor(sub_tensor_300, view_default_264);  sub_tensor_300 = view_default_264 = None
        sub_tensor_301 = torch.ops.aten.sub.Tensor(slice_tensor_44, mul_tensor_856);  slice_tensor_44 = mul_tensor_856 = None
        sub_tensor_302 = torch.ops.aten.sub.Tensor(sub_tensor_301, view_default_258);  sub_tensor_301 = view_default_258 = None
        mul_tensor_857 = torch.ops.aten.mul.Tensor(sub_tensor_302, view_default_265);  sub_tensor_302 = view_default_265 = None
        mul_tensor_858 = torch.ops.aten.mul.Tensor(sum_dim_int_list_163, arg1428_1);  sum_dim_int_list_163 = arg1428_1 = None
        convolution_backward_default_163 = torch.ops.aten.convolution_backward.default(mul_tensor_857, arg1426_1, arg526_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_857 = arg1426_1 = arg526_1 = None
        getitem_489 = convolution_backward_default_163[0]
        getitem_490 = convolution_backward_default_163[1];  convolution_backward_default_163 = None
        convolution_backward_default_164 = torch.ops.aten.convolution_backward.default(getitem_489, arg1425_1, arg525_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_489 = arg525_1 = None
        getitem_492 = convolution_backward_default_164[0]
        getitem_493 = convolution_backward_default_164[1];  convolution_backward_default_164 = None
        le_scalar_60 = torch.ops.aten.le.Scalar(arg1425_1, 0);  arg1425_1 = None
        new_zeros_default_89 = torch.ops.aten.new_zeros.default(getitem_492, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_89 = torch.ops.aten.where.self(le_scalar_60, new_zeros_default_89, getitem_492);  le_scalar_60 = new_zeros_default_89 = getitem_492 = None
        sum_dim_int_list_164 = torch.ops.aten.sum.dim_IntList(where_self_89, [0, 2, 3])
        sub_tensor_303 = torch.ops.aten.sub.Tensor(arg1423_1, arg1827_1);  arg1423_1 = arg1827_1 = None
        mul_tensor_859 = torch.ops.aten.mul.Tensor(where_self_89, sub_tensor_303)
        sum_dim_int_list_165 = torch.ops.aten.sum.dim_IntList(mul_tensor_859, [0, 2, 3]);  mul_tensor_859 = None
        mul_tensor_860 = torch.ops.aten.mul.Tensor(sum_dim_int_list_164, 0.0022675736961451248);  sum_dim_int_list_164 = None
        view_default_266 = torch.ops.aten.view.default(mul_tensor_860, [1, 336, 1, 1]);  mul_tensor_860 = None
        mul_tensor_861 = torch.ops.aten.mul.Tensor(sum_dim_int_list_165, 0.0022675736961451248)
        mul_tensor_862 = torch.ops.aten.mul.Tensor(arg1424_1, arg1424_1)
        mul_tensor_863 = torch.ops.aten.mul.Tensor(mul_tensor_861, mul_tensor_862);  mul_tensor_861 = mul_tensor_862 = None
        view_default_267 = torch.ops.aten.view.default(mul_tensor_863, [1, 336, 1, 1]);  mul_tensor_863 = None
        mul_tensor_864 = torch.ops.aten.mul.Tensor(arg1424_1, arg524_1);  arg524_1 = None
        view_default_268 = torch.ops.aten.view.default(mul_tensor_864, [1, 336, 1, 1]);  mul_tensor_864 = None
        mul_tensor_865 = torch.ops.aten.mul.Tensor(sub_tensor_303, view_default_267);  sub_tensor_303 = view_default_267 = None
        sub_tensor_304 = torch.ops.aten.sub.Tensor(where_self_89, mul_tensor_865);  where_self_89 = mul_tensor_865 = None
        sub_tensor_305 = torch.ops.aten.sub.Tensor(sub_tensor_304, view_default_266);  sub_tensor_304 = view_default_266 = None
        mul_tensor_866 = torch.ops.aten.mul.Tensor(sub_tensor_305, view_default_268);  sub_tensor_305 = view_default_268 = None
        mul_tensor_867 = torch.ops.aten.mul.Tensor(sum_dim_int_list_165, arg1424_1);  sum_dim_int_list_165 = arg1424_1 = None
        convolution_backward_default_165 = torch.ops.aten.convolution_backward.default(mul_tensor_866, arg1422_1, arg523_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_866 = arg1422_1 = arg523_1 = None
        getitem_495 = convolution_backward_default_165[0]
        getitem_496 = convolution_backward_default_165[1];  convolution_backward_default_165 = None
        convolution_backward_default_166 = torch.ops.aten.convolution_backward.default(getitem_495, relu_default_26, arg522_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_495 = arg522_1 = None
        getitem_498 = convolution_backward_default_166[0]
        getitem_499 = convolution_backward_default_166[1];  convolution_backward_default_166 = None
        new_zeros_default_90 = torch.ops.aten.new_zeros.default(getitem_498, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_90 = torch.ops.aten.where.self(le_scalar_59, new_zeros_default_90, getitem_498);  new_zeros_default_90 = getitem_498 = None
        add_tensor_110 = torch.ops.aten.add.Tensor(add_tensor_109, where_self_90);  add_tensor_109 = where_self_90 = None
        sum_dim_int_list_166 = torch.ops.aten.sum.dim_IntList(slice_tensor_43, [0, 2, 3])
        sub_tensor_306 = torch.ops.aten.sub.Tensor(arg1420_1, arg1828_1);  arg1420_1 = arg1828_1 = None
        mul_tensor_868 = torch.ops.aten.mul.Tensor(slice_tensor_43, sub_tensor_306)
        sum_dim_int_list_167 = torch.ops.aten.sum.dim_IntList(mul_tensor_868, [0, 2, 3]);  mul_tensor_868 = None
        mul_tensor_869 = torch.ops.aten.mul.Tensor(sum_dim_int_list_166, 0.0022675736961451248);  sum_dim_int_list_166 = None
        view_default_269 = torch.ops.aten.view.default(mul_tensor_869, [1, 336, 1, 1]);  mul_tensor_869 = None
        mul_tensor_870 = torch.ops.aten.mul.Tensor(sum_dim_int_list_167, 0.0022675736961451248)
        mul_tensor_871 = torch.ops.aten.mul.Tensor(arg1421_1, arg1421_1)
        mul_tensor_872 = torch.ops.aten.mul.Tensor(mul_tensor_870, mul_tensor_871);  mul_tensor_870 = mul_tensor_871 = None
        view_default_270 = torch.ops.aten.view.default(mul_tensor_872, [1, 336, 1, 1]);  mul_tensor_872 = None
        mul_tensor_873 = torch.ops.aten.mul.Tensor(arg1421_1, arg521_1);  arg521_1 = None
        view_default_271 = torch.ops.aten.view.default(mul_tensor_873, [1, 336, 1, 1]);  mul_tensor_873 = None
        mul_tensor_874 = torch.ops.aten.mul.Tensor(sub_tensor_306, view_default_270);  sub_tensor_306 = view_default_270 = None
        sub_tensor_307 = torch.ops.aten.sub.Tensor(slice_tensor_43, mul_tensor_874);  mul_tensor_874 = None
        sub_tensor_308 = torch.ops.aten.sub.Tensor(sub_tensor_307, view_default_269);  sub_tensor_307 = None
        mul_tensor_875 = torch.ops.aten.mul.Tensor(sub_tensor_308, view_default_271);  sub_tensor_308 = view_default_271 = None
        mul_tensor_876 = torch.ops.aten.mul.Tensor(sum_dim_int_list_167, arg1421_1);  sum_dim_int_list_167 = arg1421_1 = None
        convolution_backward_default_167 = torch.ops.aten.convolution_backward.default(mul_tensor_875, arg1419_1, arg520_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_875 = arg1419_1 = arg520_1 = None
        getitem_501 = convolution_backward_default_167[0]
        getitem_502 = convolution_backward_default_167[1];  convolution_backward_default_167 = None
        convolution_backward_default_168 = torch.ops.aten.convolution_backward.default(getitem_501, arg1418_1, arg519_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_501 = arg519_1 = None
        getitem_504 = convolution_backward_default_168[0]
        getitem_505 = convolution_backward_default_168[1];  convolution_backward_default_168 = None
        le_scalar_61 = torch.ops.aten.le.Scalar(arg1418_1, 0);  arg1418_1 = None
        new_zeros_default_91 = torch.ops.aten.new_zeros.default(getitem_504, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_91 = torch.ops.aten.where.self(le_scalar_61, new_zeros_default_91, getitem_504);  le_scalar_61 = new_zeros_default_91 = getitem_504 = None
        sum_dim_int_list_168 = torch.ops.aten.sum.dim_IntList(where_self_91, [0, 2, 3])
        sub_tensor_309 = torch.ops.aten.sub.Tensor(arg1416_1, arg1829_1);  arg1416_1 = arg1829_1 = None
        mul_tensor_877 = torch.ops.aten.mul.Tensor(where_self_91, sub_tensor_309)
        sum_dim_int_list_169 = torch.ops.aten.sum.dim_IntList(mul_tensor_877, [0, 2, 3]);  mul_tensor_877 = None
        mul_tensor_878 = torch.ops.aten.mul.Tensor(sum_dim_int_list_168, 0.0022675736961451248);  sum_dim_int_list_168 = None
        view_default_272 = torch.ops.aten.view.default(mul_tensor_878, [1, 336, 1, 1]);  mul_tensor_878 = None
        mul_tensor_879 = torch.ops.aten.mul.Tensor(sum_dim_int_list_169, 0.0022675736961451248)
        mul_tensor_880 = torch.ops.aten.mul.Tensor(arg1417_1, arg1417_1)
        mul_tensor_881 = torch.ops.aten.mul.Tensor(mul_tensor_879, mul_tensor_880);  mul_tensor_879 = mul_tensor_880 = None
        view_default_273 = torch.ops.aten.view.default(mul_tensor_881, [1, 336, 1, 1]);  mul_tensor_881 = None
        mul_tensor_882 = torch.ops.aten.mul.Tensor(arg1417_1, arg518_1);  arg518_1 = None
        view_default_274 = torch.ops.aten.view.default(mul_tensor_882, [1, 336, 1, 1]);  mul_tensor_882 = None
        mul_tensor_883 = torch.ops.aten.mul.Tensor(sub_tensor_309, view_default_273);  sub_tensor_309 = view_default_273 = None
        sub_tensor_310 = torch.ops.aten.sub.Tensor(where_self_91, mul_tensor_883);  where_self_91 = mul_tensor_883 = None
        sub_tensor_311 = torch.ops.aten.sub.Tensor(sub_tensor_310, view_default_272);  sub_tensor_310 = view_default_272 = None
        mul_tensor_884 = torch.ops.aten.mul.Tensor(sub_tensor_311, view_default_274);  sub_tensor_311 = view_default_274 = None
        mul_tensor_885 = torch.ops.aten.mul.Tensor(sum_dim_int_list_169, arg1417_1);  sum_dim_int_list_169 = arg1417_1 = None
        convolution_backward_default_169 = torch.ops.aten.convolution_backward.default(mul_tensor_884, arg1415_1, arg517_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_884 = arg1415_1 = arg517_1 = None
        getitem_507 = convolution_backward_default_169[0]
        getitem_508 = convolution_backward_default_169[1];  convolution_backward_default_169 = None
        convolution_backward_default_170 = torch.ops.aten.convolution_backward.default(getitem_507, relu_default_26, arg516_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_507 = relu_default_26 = arg516_1 = None
        getitem_510 = convolution_backward_default_170[0]
        getitem_511 = convolution_backward_default_170[1];  convolution_backward_default_170 = None
        new_zeros_default_92 = torch.ops.aten.new_zeros.default(getitem_510, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_92 = torch.ops.aten.where.self(le_scalar_59, new_zeros_default_92, getitem_510);  le_scalar_59 = new_zeros_default_92 = getitem_510 = None
        add_tensor_111 = torch.ops.aten.add.Tensor(add_tensor_110, where_self_92);  add_tensor_110 = where_self_92 = None
        sub_tensor_312 = torch.ops.aten.sub.Tensor(arg1413_1, arg1830_1);  arg1413_1 = arg1830_1 = None
        mul_tensor_886 = torch.ops.aten.mul.Tensor(slice_tensor_43, sub_tensor_312)
        sum_dim_int_list_170 = torch.ops.aten.sum.dim_IntList(mul_tensor_886, [0, 2, 3]);  mul_tensor_886 = None
        mul_tensor_887 = torch.ops.aten.mul.Tensor(sum_dim_int_list_170, 0.0022675736961451248)
        mul_tensor_888 = torch.ops.aten.mul.Tensor(arg1414_1, arg1414_1)
        mul_tensor_889 = torch.ops.aten.mul.Tensor(mul_tensor_887, mul_tensor_888);  mul_tensor_887 = mul_tensor_888 = None
        view_default_275 = torch.ops.aten.view.default(mul_tensor_889, [1, 336, 1, 1]);  mul_tensor_889 = None
        mul_tensor_890 = torch.ops.aten.mul.Tensor(arg1414_1, arg515_1);  arg515_1 = None
        view_default_276 = torch.ops.aten.view.default(mul_tensor_890, [1, 336, 1, 1]);  mul_tensor_890 = None
        mul_tensor_891 = torch.ops.aten.mul.Tensor(sub_tensor_312, view_default_275);  sub_tensor_312 = view_default_275 = None
        sub_tensor_313 = torch.ops.aten.sub.Tensor(slice_tensor_43, mul_tensor_891);  slice_tensor_43 = mul_tensor_891 = None
        sub_tensor_314 = torch.ops.aten.sub.Tensor(sub_tensor_313, view_default_269);  sub_tensor_313 = view_default_269 = None
        mul_tensor_892 = torch.ops.aten.mul.Tensor(sub_tensor_314, view_default_276);  sub_tensor_314 = view_default_276 = None
        mul_tensor_893 = torch.ops.aten.mul.Tensor(sum_dim_int_list_170, arg1414_1);  sum_dim_int_list_170 = arg1414_1 = None
        convolution_backward_default_171 = torch.ops.aten.convolution_backward.default(mul_tensor_892, arg1412_1, arg514_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_892 = arg1412_1 = arg514_1 = None
        getitem_513 = convolution_backward_default_171[0]
        getitem_514 = convolution_backward_default_171[1];  convolution_backward_default_171 = None
        convolution_backward_default_172 = torch.ops.aten.convolution_backward.default(getitem_513, arg1411_1, arg513_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_513 = arg513_1 = None
        getitem_516 = convolution_backward_default_172[0]
        getitem_517 = convolution_backward_default_172[1];  convolution_backward_default_172 = None
        le_scalar_62 = torch.ops.aten.le.Scalar(arg1411_1, 0);  arg1411_1 = None
        new_zeros_default_93 = torch.ops.aten.new_zeros.default(getitem_516, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_93 = torch.ops.aten.where.self(le_scalar_62, new_zeros_default_93, getitem_516);  le_scalar_62 = new_zeros_default_93 = getitem_516 = None
        sum_dim_int_list_171 = torch.ops.aten.sum.dim_IntList(where_self_93, [0, 2, 3])
        sub_tensor_315 = torch.ops.aten.sub.Tensor(arg1409_1, arg1831_1);  arg1409_1 = arg1831_1 = None
        mul_tensor_894 = torch.ops.aten.mul.Tensor(where_self_93, sub_tensor_315)
        sum_dim_int_list_172 = torch.ops.aten.sum.dim_IntList(mul_tensor_894, [0, 2, 3]);  mul_tensor_894 = None
        mul_tensor_895 = torch.ops.aten.mul.Tensor(sum_dim_int_list_171, 0.0022675736961451248);  sum_dim_int_list_171 = None
        view_default_277 = torch.ops.aten.view.default(mul_tensor_895, [1, 336, 1, 1]);  mul_tensor_895 = None
        mul_tensor_896 = torch.ops.aten.mul.Tensor(sum_dim_int_list_172, 0.0022675736961451248)
        mul_tensor_897 = torch.ops.aten.mul.Tensor(arg1410_1, arg1410_1)
        mul_tensor_898 = torch.ops.aten.mul.Tensor(mul_tensor_896, mul_tensor_897);  mul_tensor_896 = mul_tensor_897 = None
        view_default_278 = torch.ops.aten.view.default(mul_tensor_898, [1, 336, 1, 1]);  mul_tensor_898 = None
        mul_tensor_899 = torch.ops.aten.mul.Tensor(arg1410_1, arg512_1);  arg512_1 = None
        view_default_279 = torch.ops.aten.view.default(mul_tensor_899, [1, 336, 1, 1]);  mul_tensor_899 = None
        mul_tensor_900 = torch.ops.aten.mul.Tensor(sub_tensor_315, view_default_278);  sub_tensor_315 = view_default_278 = None
        sub_tensor_316 = torch.ops.aten.sub.Tensor(where_self_93, mul_tensor_900);  where_self_93 = mul_tensor_900 = None
        sub_tensor_317 = torch.ops.aten.sub.Tensor(sub_tensor_316, view_default_277);  sub_tensor_316 = view_default_277 = None
        mul_tensor_901 = torch.ops.aten.mul.Tensor(sub_tensor_317, view_default_279);  sub_tensor_317 = view_default_279 = None
        mul_tensor_902 = torch.ops.aten.mul.Tensor(sum_dim_int_list_172, arg1410_1);  sum_dim_int_list_172 = arg1410_1 = None
        convolution_backward_default_173 = torch.ops.aten.convolution_backward.default(mul_tensor_901, arg1408_1, arg511_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_901 = arg1408_1 = arg511_1 = None
        getitem_519 = convolution_backward_default_173[0]
        getitem_520 = convolution_backward_default_173[1];  convolution_backward_default_173 = None
        convolution_backward_default_174 = torch.ops.aten.convolution_backward.default(getitem_519, relu_default_25, arg510_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_519 = relu_default_25 = arg510_1 = None
        getitem_522 = convolution_backward_default_174[0]
        getitem_523 = convolution_backward_default_174[1];  convolution_backward_default_174 = None
        new_zeros_default_94 = torch.ops.aten.new_zeros.default(getitem_522, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_94 = torch.ops.aten.where.self(le_scalar_57, new_zeros_default_94, getitem_522);  le_scalar_57 = new_zeros_default_94 = getitem_522 = None
        add_tensor_112 = torch.ops.aten.add.Tensor(add_tensor_108, where_self_94);  add_tensor_108 = where_self_94 = None
        view_default_280 = torch.ops.aten.view.default(squeeze_dim_140, [1, 336, 1, 1]);  squeeze_dim_140 = None
        sum_dim_int_list_173 = torch.ops.aten.sum.dim_IntList(add_tensor_112, [0, 2, 3])
        sub_tensor_318 = torch.ops.aten.sub.Tensor(arg1405_1, view_default_280);  arg1405_1 = view_default_280 = None
        mul_tensor_903 = torch.ops.aten.mul.Tensor(add_tensor_112, sub_tensor_318)
        sum_dim_int_list_174 = torch.ops.aten.sum.dim_IntList(mul_tensor_903, [0, 2, 3]);  mul_tensor_903 = None
        mul_tensor_904 = torch.ops.aten.mul.Tensor(sum_dim_int_list_173, 0.0022675736961451248);  sum_dim_int_list_173 = None
        view_default_281 = torch.ops.aten.view.default(mul_tensor_904, [1, 336, 1, 1]);  mul_tensor_904 = None
        mul_tensor_905 = torch.ops.aten.mul.Tensor(sum_dim_int_list_174, 0.0022675736961451248)
        mul_tensor_906 = torch.ops.aten.mul.Tensor(squeeze_dim_143, squeeze_dim_143)
        mul_tensor_907 = torch.ops.aten.mul.Tensor(mul_tensor_905, mul_tensor_906);  mul_tensor_905 = mul_tensor_906 = None
        view_default_282 = torch.ops.aten.view.default(mul_tensor_907, [1, 336, 1, 1]);  mul_tensor_907 = None
        mul_tensor_908 = torch.ops.aten.mul.Tensor(squeeze_dim_143, arg508_1);  arg508_1 = None
        view_default_283 = torch.ops.aten.view.default(mul_tensor_908, [1, 336, 1, 1]);  mul_tensor_908 = None
        mul_tensor_909 = torch.ops.aten.mul.Tensor(sub_tensor_318, view_default_282);  sub_tensor_318 = view_default_282 = None
        sub_tensor_319 = torch.ops.aten.sub.Tensor(add_tensor_112, mul_tensor_909);  add_tensor_112 = mul_tensor_909 = None
        sub_tensor_320 = torch.ops.aten.sub.Tensor(sub_tensor_319, view_default_281);  sub_tensor_319 = view_default_281 = None
        mul_tensor_910 = torch.ops.aten.mul.Tensor(sub_tensor_320, view_default_283);  sub_tensor_320 = view_default_283 = None
        mul_tensor_911 = torch.ops.aten.mul.Tensor(sum_dim_int_list_174, squeeze_dim_143);  sum_dim_int_list_174 = squeeze_dim_143 = None
        convolution_backward_default_175 = torch.ops.aten.convolution_backward.default(mul_tensor_910, arg1404_1, arg507_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_910 = arg1404_1 = arg507_1 = None
        getitem_525 = convolution_backward_default_175[0]
        getitem_526 = convolution_backward_default_175[1];  convolution_backward_default_175 = None
        new_zeros_default_95 = torch.ops.aten.new_zeros.default(getitem_525, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_95 = torch.ops.aten.where.self(le_scalar_48, new_zeros_default_95, getitem_525);  le_scalar_48 = new_zeros_default_95 = getitem_525 = None
        add_tensor_113 = torch.ops.aten.add.Tensor(add_tensor_103, where_self_95);  add_tensor_103 = where_self_95 = None
        view_default_284 = torch.ops.aten.view.default(squeeze_dim_134, [1, 336, 1, 1]);  squeeze_dim_134 = None
        sum_dim_int_list_175 = torch.ops.aten.sum.dim_IntList(add_tensor_111, [0, 2, 3])
        sub_tensor_321 = torch.ops.aten.sub.Tensor(arg1401_1, view_default_284);  arg1401_1 = view_default_284 = None
        mul_tensor_912 = torch.ops.aten.mul.Tensor(add_tensor_111, sub_tensor_321)
        sum_dim_int_list_176 = torch.ops.aten.sum.dim_IntList(mul_tensor_912, [0, 2, 3]);  mul_tensor_912 = None
        mul_tensor_913 = torch.ops.aten.mul.Tensor(sum_dim_int_list_175, 0.0022675736961451248);  sum_dim_int_list_175 = None
        view_default_285 = torch.ops.aten.view.default(mul_tensor_913, [1, 336, 1, 1]);  mul_tensor_913 = None
        mul_tensor_914 = torch.ops.aten.mul.Tensor(sum_dim_int_list_176, 0.0022675736961451248)
        mul_tensor_915 = torch.ops.aten.mul.Tensor(squeeze_dim_137, squeeze_dim_137)
        mul_tensor_916 = torch.ops.aten.mul.Tensor(mul_tensor_914, mul_tensor_915);  mul_tensor_914 = mul_tensor_915 = None
        view_default_286 = torch.ops.aten.view.default(mul_tensor_916, [1, 336, 1, 1]);  mul_tensor_916 = None
        mul_tensor_917 = torch.ops.aten.mul.Tensor(squeeze_dim_137, arg505_1);  arg505_1 = None
        view_default_287 = torch.ops.aten.view.default(mul_tensor_917, [1, 336, 1, 1]);  mul_tensor_917 = None
        mul_tensor_918 = torch.ops.aten.mul.Tensor(sub_tensor_321, view_default_286);  sub_tensor_321 = view_default_286 = None
        sub_tensor_322 = torch.ops.aten.sub.Tensor(add_tensor_111, mul_tensor_918);  add_tensor_111 = mul_tensor_918 = None
        sub_tensor_323 = torch.ops.aten.sub.Tensor(sub_tensor_322, view_default_285);  sub_tensor_322 = view_default_285 = None
        mul_tensor_919 = torch.ops.aten.mul.Tensor(sub_tensor_323, view_default_287);  sub_tensor_323 = view_default_287 = None
        mul_tensor_920 = torch.ops.aten.mul.Tensor(sum_dim_int_list_176, squeeze_dim_137);  sum_dim_int_list_176 = squeeze_dim_137 = None
        convolution_backward_default_176 = torch.ops.aten.convolution_backward.default(mul_tensor_919, arg1362_1, arg504_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_919 = arg504_1 = None
        getitem_528 = convolution_backward_default_176[0]
        getitem_529 = convolution_backward_default_176[1];  convolution_backward_default_176 = None
        le_scalar_63 = torch.ops.aten.le.Scalar(arg1362_1, 0)
        new_zeros_default_96 = torch.ops.aten.new_zeros.default(getitem_528, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_96 = torch.ops.aten.where.self(le_scalar_63, new_zeros_default_96, getitem_528);  new_zeros_default_96 = getitem_528 = None
        slice_tensor_48 = torch.ops.aten.slice.Tensor(add_tensor_113, 1, 0, 336)
        slice_tensor_49 = torch.ops.aten.slice.Tensor(add_tensor_113, 1, 336, 672)
        slice_tensor_50 = torch.ops.aten.slice.Tensor(add_tensor_113, 1, 672, 1008)
        slice_tensor_51 = torch.ops.aten.slice.Tensor(add_tensor_113, 1, 1008, 1344)
        slice_tensor_52 = torch.ops.aten.slice.Tensor(add_tensor_113, 1, 1344, 1680)
        slice_tensor_53 = torch.ops.aten.slice.Tensor(add_tensor_113, 1, 1680, 2016);  add_tensor_113 = None
        sum_dim_int_list_177 = torch.ops.aten.sum.dim_IntList(slice_tensor_53, [0, 2, 3])
        sub_tensor_324 = torch.ops.aten.sub.Tensor(arg1399_1, arg1832_1);  arg1399_1 = arg1832_1 = None
        mul_tensor_921 = torch.ops.aten.mul.Tensor(slice_tensor_53, sub_tensor_324)
        sum_dim_int_list_178 = torch.ops.aten.sum.dim_IntList(mul_tensor_921, [0, 2, 3]);  mul_tensor_921 = None
        mul_tensor_922 = torch.ops.aten.mul.Tensor(sum_dim_int_list_177, 0.0022675736961451248);  sum_dim_int_list_177 = None
        view_default_288 = torch.ops.aten.view.default(mul_tensor_922, [1, 336, 1, 1]);  mul_tensor_922 = None
        mul_tensor_923 = torch.ops.aten.mul.Tensor(sum_dim_int_list_178, 0.0022675736961451248)
        mul_tensor_924 = torch.ops.aten.mul.Tensor(arg1400_1, arg1400_1)
        mul_tensor_925 = torch.ops.aten.mul.Tensor(mul_tensor_923, mul_tensor_924);  mul_tensor_923 = mul_tensor_924 = None
        view_default_289 = torch.ops.aten.view.default(mul_tensor_925, [1, 336, 1, 1]);  mul_tensor_925 = None
        mul_tensor_926 = torch.ops.aten.mul.Tensor(arg1400_1, arg503_1);  arg503_1 = None
        view_default_290 = torch.ops.aten.view.default(mul_tensor_926, [1, 336, 1, 1]);  mul_tensor_926 = None
        mul_tensor_927 = torch.ops.aten.mul.Tensor(sub_tensor_324, view_default_289);  sub_tensor_324 = view_default_289 = None
        sub_tensor_325 = torch.ops.aten.sub.Tensor(slice_tensor_53, mul_tensor_927);  mul_tensor_927 = None
        sub_tensor_326 = torch.ops.aten.sub.Tensor(sub_tensor_325, view_default_288);  sub_tensor_325 = view_default_288 = None
        mul_tensor_928 = torch.ops.aten.mul.Tensor(sub_tensor_326, view_default_290);  sub_tensor_326 = view_default_290 = None
        mul_tensor_929 = torch.ops.aten.mul.Tensor(sum_dim_int_list_178, arg1400_1);  sum_dim_int_list_178 = arg1400_1 = None
        convolution_backward_default_177 = torch.ops.aten.convolution_backward.default(mul_tensor_928, arg1398_1, arg502_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_928 = arg1398_1 = arg502_1 = None
        getitem_531 = convolution_backward_default_177[0]
        getitem_532 = convolution_backward_default_177[1];  convolution_backward_default_177 = None
        convolution_backward_default_178 = torch.ops.aten.convolution_backward.default(getitem_531, arg1397_1, arg501_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_531 = arg501_1 = None
        getitem_534 = convolution_backward_default_178[0]
        getitem_535 = convolution_backward_default_178[1];  convolution_backward_default_178 = None
        le_scalar_64 = torch.ops.aten.le.Scalar(arg1397_1, 0);  arg1397_1 = None
        new_zeros_default_97 = torch.ops.aten.new_zeros.default(getitem_534, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_97 = torch.ops.aten.where.self(le_scalar_64, new_zeros_default_97, getitem_534);  le_scalar_64 = new_zeros_default_97 = getitem_534 = None
        sum_dim_int_list_179 = torch.ops.aten.sum.dim_IntList(where_self_97, [0, 2, 3])
        sub_tensor_327 = torch.ops.aten.sub.Tensor(arg1395_1, arg1833_1);  arg1395_1 = arg1833_1 = None
        mul_tensor_930 = torch.ops.aten.mul.Tensor(where_self_97, sub_tensor_327)
        sum_dim_int_list_180 = torch.ops.aten.sum.dim_IntList(mul_tensor_930, [0, 2, 3]);  mul_tensor_930 = None
        mul_tensor_931 = torch.ops.aten.mul.Tensor(sum_dim_int_list_179, 0.0022675736961451248);  sum_dim_int_list_179 = None
        view_default_291 = torch.ops.aten.view.default(mul_tensor_931, [1, 336, 1, 1]);  mul_tensor_931 = None
        mul_tensor_932 = torch.ops.aten.mul.Tensor(sum_dim_int_list_180, 0.0022675736961451248)
        mul_tensor_933 = torch.ops.aten.mul.Tensor(arg1396_1, arg1396_1)
        mul_tensor_934 = torch.ops.aten.mul.Tensor(mul_tensor_932, mul_tensor_933);  mul_tensor_932 = mul_tensor_933 = None
        view_default_292 = torch.ops.aten.view.default(mul_tensor_934, [1, 336, 1, 1]);  mul_tensor_934 = None
        mul_tensor_935 = torch.ops.aten.mul.Tensor(arg1396_1, arg500_1);  arg500_1 = None
        view_default_293 = torch.ops.aten.view.default(mul_tensor_935, [1, 336, 1, 1]);  mul_tensor_935 = None
        mul_tensor_936 = torch.ops.aten.mul.Tensor(sub_tensor_327, view_default_292);  sub_tensor_327 = view_default_292 = None
        sub_tensor_328 = torch.ops.aten.sub.Tensor(where_self_97, mul_tensor_936);  where_self_97 = mul_tensor_936 = None
        sub_tensor_329 = torch.ops.aten.sub.Tensor(sub_tensor_328, view_default_291);  sub_tensor_328 = view_default_291 = None
        mul_tensor_937 = torch.ops.aten.mul.Tensor(sub_tensor_329, view_default_293);  sub_tensor_329 = view_default_293 = None
        mul_tensor_938 = torch.ops.aten.mul.Tensor(sum_dim_int_list_180, arg1396_1);  sum_dim_int_list_180 = arg1396_1 = None
        convolution_backward_default_179 = torch.ops.aten.convolution_backward.default(mul_tensor_937, arg1394_1, arg499_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_937 = arg1394_1 = arg499_1 = None
        getitem_537 = convolution_backward_default_179[0]
        getitem_538 = convolution_backward_default_179[1];  convolution_backward_default_179 = None
        convolution_backward_default_180 = torch.ops.aten.convolution_backward.default(getitem_537, relu_default_23, arg498_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_537 = arg498_1 = None
        getitem_540 = convolution_backward_default_180[0]
        getitem_541 = convolution_backward_default_180[1];  convolution_backward_default_180 = None
        le_scalar_65 = torch.ops.aten.le.Scalar(relu_default_23, 0)
        new_zeros_default_98 = torch.ops.aten.new_zeros.default(getitem_540, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_98 = torch.ops.aten.where.self(le_scalar_65, new_zeros_default_98, getitem_540);  new_zeros_default_98 = getitem_540 = None
        add_tensor_114 = torch.ops.aten.add.Tensor(slice_tensor_53, where_self_98);  slice_tensor_53 = where_self_98 = None
        avg_pool2d_backward_default_18 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_52, add_tensor_20, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_52 = add_tensor_20 = None
        add_tensor_115 = torch.ops.aten.add.Tensor(slice_tensor_48, avg_pool2d_backward_default_18);  slice_tensor_48 = None
        add_tensor_116 = torch.ops.aten.add.Tensor(add_tensor_115, avg_pool2d_backward_default_18);  add_tensor_115 = avg_pool2d_backward_default_18 = None
        add_tensor_117 = torch.ops.aten.add.Tensor(add_tensor_116, slice_tensor_51);  add_tensor_116 = None
        avg_pool2d_backward_default_19 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_51, add_tensor_21, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_51 = add_tensor_21 = None
        add_tensor_118 = torch.ops.aten.add.Tensor(add_tensor_114, avg_pool2d_backward_default_19);  add_tensor_114 = avg_pool2d_backward_default_19 = None
        sum_dim_int_list_181 = torch.ops.aten.sum.dim_IntList(slice_tensor_50, [0, 2, 3])
        sub_tensor_330 = torch.ops.aten.sub.Tensor(arg1392_1, arg1834_1);  arg1392_1 = arg1834_1 = None
        mul_tensor_939 = torch.ops.aten.mul.Tensor(slice_tensor_50, sub_tensor_330)
        sum_dim_int_list_182 = torch.ops.aten.sum.dim_IntList(mul_tensor_939, [0, 2, 3]);  mul_tensor_939 = None
        mul_tensor_940 = torch.ops.aten.mul.Tensor(sum_dim_int_list_181, 0.0022675736961451248);  sum_dim_int_list_181 = None
        view_default_294 = torch.ops.aten.view.default(mul_tensor_940, [1, 336, 1, 1]);  mul_tensor_940 = None
        mul_tensor_941 = torch.ops.aten.mul.Tensor(sum_dim_int_list_182, 0.0022675736961451248)
        mul_tensor_942 = torch.ops.aten.mul.Tensor(arg1393_1, arg1393_1)
        mul_tensor_943 = torch.ops.aten.mul.Tensor(mul_tensor_941, mul_tensor_942);  mul_tensor_941 = mul_tensor_942 = None
        view_default_295 = torch.ops.aten.view.default(mul_tensor_943, [1, 336, 1, 1]);  mul_tensor_943 = None
        mul_tensor_944 = torch.ops.aten.mul.Tensor(arg1393_1, arg497_1);  arg497_1 = None
        view_default_296 = torch.ops.aten.view.default(mul_tensor_944, [1, 336, 1, 1]);  mul_tensor_944 = None
        mul_tensor_945 = torch.ops.aten.mul.Tensor(sub_tensor_330, view_default_295);  sub_tensor_330 = view_default_295 = None
        sub_tensor_331 = torch.ops.aten.sub.Tensor(slice_tensor_50, mul_tensor_945);  mul_tensor_945 = None
        sub_tensor_332 = torch.ops.aten.sub.Tensor(sub_tensor_331, view_default_294);  sub_tensor_331 = None
        mul_tensor_946 = torch.ops.aten.mul.Tensor(sub_tensor_332, view_default_296);  sub_tensor_332 = view_default_296 = None
        mul_tensor_947 = torch.ops.aten.mul.Tensor(sum_dim_int_list_182, arg1393_1);  sum_dim_int_list_182 = arg1393_1 = None
        convolution_backward_default_181 = torch.ops.aten.convolution_backward.default(mul_tensor_946, arg1391_1, arg496_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_946 = arg1391_1 = arg496_1 = None
        getitem_543 = convolution_backward_default_181[0]
        getitem_544 = convolution_backward_default_181[1];  convolution_backward_default_181 = None
        convolution_backward_default_182 = torch.ops.aten.convolution_backward.default(getitem_543, arg1390_1, arg495_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_543 = arg495_1 = None
        getitem_546 = convolution_backward_default_182[0]
        getitem_547 = convolution_backward_default_182[1];  convolution_backward_default_182 = None
        le_scalar_66 = torch.ops.aten.le.Scalar(arg1390_1, 0);  arg1390_1 = None
        new_zeros_default_99 = torch.ops.aten.new_zeros.default(getitem_546, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_99 = torch.ops.aten.where.self(le_scalar_66, new_zeros_default_99, getitem_546);  le_scalar_66 = new_zeros_default_99 = getitem_546 = None
        sum_dim_int_list_183 = torch.ops.aten.sum.dim_IntList(where_self_99, [0, 2, 3])
        sub_tensor_333 = torch.ops.aten.sub.Tensor(arg1388_1, arg1835_1);  arg1388_1 = arg1835_1 = None
        mul_tensor_948 = torch.ops.aten.mul.Tensor(where_self_99, sub_tensor_333)
        sum_dim_int_list_184 = torch.ops.aten.sum.dim_IntList(mul_tensor_948, [0, 2, 3]);  mul_tensor_948 = None
        mul_tensor_949 = torch.ops.aten.mul.Tensor(sum_dim_int_list_183, 0.0022675736961451248);  sum_dim_int_list_183 = None
        view_default_297 = torch.ops.aten.view.default(mul_tensor_949, [1, 336, 1, 1]);  mul_tensor_949 = None
        mul_tensor_950 = torch.ops.aten.mul.Tensor(sum_dim_int_list_184, 0.0022675736961451248)
        mul_tensor_951 = torch.ops.aten.mul.Tensor(arg1389_1, arg1389_1)
        mul_tensor_952 = torch.ops.aten.mul.Tensor(mul_tensor_950, mul_tensor_951);  mul_tensor_950 = mul_tensor_951 = None
        view_default_298 = torch.ops.aten.view.default(mul_tensor_952, [1, 336, 1, 1]);  mul_tensor_952 = None
        mul_tensor_953 = torch.ops.aten.mul.Tensor(arg1389_1, arg494_1);  arg494_1 = None
        view_default_299 = torch.ops.aten.view.default(mul_tensor_953, [1, 336, 1, 1]);  mul_tensor_953 = None
        mul_tensor_954 = torch.ops.aten.mul.Tensor(sub_tensor_333, view_default_298);  sub_tensor_333 = view_default_298 = None
        sub_tensor_334 = torch.ops.aten.sub.Tensor(where_self_99, mul_tensor_954);  where_self_99 = mul_tensor_954 = None
        sub_tensor_335 = torch.ops.aten.sub.Tensor(sub_tensor_334, view_default_297);  sub_tensor_334 = view_default_297 = None
        mul_tensor_955 = torch.ops.aten.mul.Tensor(sub_tensor_335, view_default_299);  sub_tensor_335 = view_default_299 = None
        mul_tensor_956 = torch.ops.aten.mul.Tensor(sum_dim_int_list_184, arg1389_1);  sum_dim_int_list_184 = arg1389_1 = None
        convolution_backward_default_183 = torch.ops.aten.convolution_backward.default(mul_tensor_955, arg1387_1, arg493_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_955 = arg1387_1 = arg493_1 = None
        getitem_549 = convolution_backward_default_183[0]
        getitem_550 = convolution_backward_default_183[1];  convolution_backward_default_183 = None
        convolution_backward_default_184 = torch.ops.aten.convolution_backward.default(getitem_549, relu_default_24, arg492_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_549 = arg492_1 = None
        getitem_552 = convolution_backward_default_184[0]
        getitem_553 = convolution_backward_default_184[1];  convolution_backward_default_184 = None
        le_scalar_67 = torch.ops.aten.le.Scalar(relu_default_24, 0)
        new_zeros_default_100 = torch.ops.aten.new_zeros.default(getitem_552, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_100 = torch.ops.aten.where.self(le_scalar_67, new_zeros_default_100, getitem_552);  new_zeros_default_100 = getitem_552 = None
        add_tensor_119 = torch.ops.aten.add.Tensor(add_tensor_117, where_self_100);  add_tensor_117 = where_self_100 = None
        sub_tensor_336 = torch.ops.aten.sub.Tensor(arg1385_1, arg1836_1);  arg1385_1 = arg1836_1 = None
        mul_tensor_957 = torch.ops.aten.mul.Tensor(slice_tensor_50, sub_tensor_336)
        sum_dim_int_list_185 = torch.ops.aten.sum.dim_IntList(mul_tensor_957, [0, 2, 3]);  mul_tensor_957 = None
        mul_tensor_958 = torch.ops.aten.mul.Tensor(sum_dim_int_list_185, 0.0022675736961451248)
        mul_tensor_959 = torch.ops.aten.mul.Tensor(arg1386_1, arg1386_1)
        mul_tensor_960 = torch.ops.aten.mul.Tensor(mul_tensor_958, mul_tensor_959);  mul_tensor_958 = mul_tensor_959 = None
        view_default_300 = torch.ops.aten.view.default(mul_tensor_960, [1, 336, 1, 1]);  mul_tensor_960 = None
        mul_tensor_961 = torch.ops.aten.mul.Tensor(arg1386_1, arg491_1);  arg491_1 = None
        view_default_301 = torch.ops.aten.view.default(mul_tensor_961, [1, 336, 1, 1]);  mul_tensor_961 = None
        mul_tensor_962 = torch.ops.aten.mul.Tensor(sub_tensor_336, view_default_300);  sub_tensor_336 = view_default_300 = None
        sub_tensor_337 = torch.ops.aten.sub.Tensor(slice_tensor_50, mul_tensor_962);  slice_tensor_50 = mul_tensor_962 = None
        sub_tensor_338 = torch.ops.aten.sub.Tensor(sub_tensor_337, view_default_294);  sub_tensor_337 = view_default_294 = None
        mul_tensor_963 = torch.ops.aten.mul.Tensor(sub_tensor_338, view_default_301);  sub_tensor_338 = view_default_301 = None
        mul_tensor_964 = torch.ops.aten.mul.Tensor(sum_dim_int_list_185, arg1386_1);  sum_dim_int_list_185 = arg1386_1 = None
        convolution_backward_default_185 = torch.ops.aten.convolution_backward.default(mul_tensor_963, arg1384_1, arg490_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_963 = arg1384_1 = arg490_1 = None
        getitem_555 = convolution_backward_default_185[0]
        getitem_556 = convolution_backward_default_185[1];  convolution_backward_default_185 = None
        convolution_backward_default_186 = torch.ops.aten.convolution_backward.default(getitem_555, arg1383_1, arg489_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_555 = arg489_1 = None
        getitem_558 = convolution_backward_default_186[0]
        getitem_559 = convolution_backward_default_186[1];  convolution_backward_default_186 = None
        le_scalar_68 = torch.ops.aten.le.Scalar(arg1383_1, 0);  arg1383_1 = None
        new_zeros_default_101 = torch.ops.aten.new_zeros.default(getitem_558, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_101 = torch.ops.aten.where.self(le_scalar_68, new_zeros_default_101, getitem_558);  le_scalar_68 = new_zeros_default_101 = getitem_558 = None
        sum_dim_int_list_186 = torch.ops.aten.sum.dim_IntList(where_self_101, [0, 2, 3])
        sub_tensor_339 = torch.ops.aten.sub.Tensor(arg1381_1, arg1837_1);  arg1381_1 = arg1837_1 = None
        mul_tensor_965 = torch.ops.aten.mul.Tensor(where_self_101, sub_tensor_339)
        sum_dim_int_list_187 = torch.ops.aten.sum.dim_IntList(mul_tensor_965, [0, 2, 3]);  mul_tensor_965 = None
        mul_tensor_966 = torch.ops.aten.mul.Tensor(sum_dim_int_list_186, 0.0022675736961451248);  sum_dim_int_list_186 = None
        view_default_302 = torch.ops.aten.view.default(mul_tensor_966, [1, 336, 1, 1]);  mul_tensor_966 = None
        mul_tensor_967 = torch.ops.aten.mul.Tensor(sum_dim_int_list_187, 0.0022675736961451248)
        mul_tensor_968 = torch.ops.aten.mul.Tensor(arg1382_1, arg1382_1)
        mul_tensor_969 = torch.ops.aten.mul.Tensor(mul_tensor_967, mul_tensor_968);  mul_tensor_967 = mul_tensor_968 = None
        view_default_303 = torch.ops.aten.view.default(mul_tensor_969, [1, 336, 1, 1]);  mul_tensor_969 = None
        mul_tensor_970 = torch.ops.aten.mul.Tensor(arg1382_1, arg488_1);  arg488_1 = None
        view_default_304 = torch.ops.aten.view.default(mul_tensor_970, [1, 336, 1, 1]);  mul_tensor_970 = None
        mul_tensor_971 = torch.ops.aten.mul.Tensor(sub_tensor_339, view_default_303);  sub_tensor_339 = view_default_303 = None
        sub_tensor_340 = torch.ops.aten.sub.Tensor(where_self_101, mul_tensor_971);  where_self_101 = mul_tensor_971 = None
        sub_tensor_341 = torch.ops.aten.sub.Tensor(sub_tensor_340, view_default_302);  sub_tensor_340 = view_default_302 = None
        mul_tensor_972 = torch.ops.aten.mul.Tensor(sub_tensor_341, view_default_304);  sub_tensor_341 = view_default_304 = None
        mul_tensor_973 = torch.ops.aten.mul.Tensor(sum_dim_int_list_187, arg1382_1);  sum_dim_int_list_187 = arg1382_1 = None
        convolution_backward_default_187 = torch.ops.aten.convolution_backward.default(mul_tensor_972, arg1380_1, arg487_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_972 = arg1380_1 = arg487_1 = None
        getitem_561 = convolution_backward_default_187[0]
        getitem_562 = convolution_backward_default_187[1];  convolution_backward_default_187 = None
        convolution_backward_default_188 = torch.ops.aten.convolution_backward.default(getitem_561, relu_default_24, arg486_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_561 = arg486_1 = None
        getitem_564 = convolution_backward_default_188[0]
        getitem_565 = convolution_backward_default_188[1];  convolution_backward_default_188 = None
        new_zeros_default_102 = torch.ops.aten.new_zeros.default(getitem_564, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_102 = torch.ops.aten.where.self(le_scalar_67, new_zeros_default_102, getitem_564);  new_zeros_default_102 = getitem_564 = None
        add_tensor_120 = torch.ops.aten.add.Tensor(add_tensor_119, where_self_102);  add_tensor_119 = where_self_102 = None
        sum_dim_int_list_188 = torch.ops.aten.sum.dim_IntList(slice_tensor_49, [0, 2, 3])
        sub_tensor_342 = torch.ops.aten.sub.Tensor(arg1378_1, arg1838_1);  arg1378_1 = arg1838_1 = None
        mul_tensor_974 = torch.ops.aten.mul.Tensor(slice_tensor_49, sub_tensor_342)
        sum_dim_int_list_189 = torch.ops.aten.sum.dim_IntList(mul_tensor_974, [0, 2, 3]);  mul_tensor_974 = None
        mul_tensor_975 = torch.ops.aten.mul.Tensor(sum_dim_int_list_188, 0.0022675736961451248);  sum_dim_int_list_188 = None
        view_default_305 = torch.ops.aten.view.default(mul_tensor_975, [1, 336, 1, 1]);  mul_tensor_975 = None
        mul_tensor_976 = torch.ops.aten.mul.Tensor(sum_dim_int_list_189, 0.0022675736961451248)
        mul_tensor_977 = torch.ops.aten.mul.Tensor(arg1379_1, arg1379_1)
        mul_tensor_978 = torch.ops.aten.mul.Tensor(mul_tensor_976, mul_tensor_977);  mul_tensor_976 = mul_tensor_977 = None
        view_default_306 = torch.ops.aten.view.default(mul_tensor_978, [1, 336, 1, 1]);  mul_tensor_978 = None
        mul_tensor_979 = torch.ops.aten.mul.Tensor(arg1379_1, arg485_1);  arg485_1 = None
        view_default_307 = torch.ops.aten.view.default(mul_tensor_979, [1, 336, 1, 1]);  mul_tensor_979 = None
        mul_tensor_980 = torch.ops.aten.mul.Tensor(sub_tensor_342, view_default_306);  sub_tensor_342 = view_default_306 = None
        sub_tensor_343 = torch.ops.aten.sub.Tensor(slice_tensor_49, mul_tensor_980);  mul_tensor_980 = None
        sub_tensor_344 = torch.ops.aten.sub.Tensor(sub_tensor_343, view_default_305);  sub_tensor_343 = None
        mul_tensor_981 = torch.ops.aten.mul.Tensor(sub_tensor_344, view_default_307);  sub_tensor_344 = view_default_307 = None
        mul_tensor_982 = torch.ops.aten.mul.Tensor(sum_dim_int_list_189, arg1379_1);  sum_dim_int_list_189 = arg1379_1 = None
        convolution_backward_default_189 = torch.ops.aten.convolution_backward.default(mul_tensor_981, arg1377_1, arg484_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_981 = arg1377_1 = arg484_1 = None
        getitem_567 = convolution_backward_default_189[0]
        getitem_568 = convolution_backward_default_189[1];  convolution_backward_default_189 = None
        convolution_backward_default_190 = torch.ops.aten.convolution_backward.default(getitem_567, arg1376_1, arg483_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_567 = arg483_1 = None
        getitem_570 = convolution_backward_default_190[0]
        getitem_571 = convolution_backward_default_190[1];  convolution_backward_default_190 = None
        le_scalar_69 = torch.ops.aten.le.Scalar(arg1376_1, 0);  arg1376_1 = None
        new_zeros_default_103 = torch.ops.aten.new_zeros.default(getitem_570, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_103 = torch.ops.aten.where.self(le_scalar_69, new_zeros_default_103, getitem_570);  le_scalar_69 = new_zeros_default_103 = getitem_570 = None
        sum_dim_int_list_190 = torch.ops.aten.sum.dim_IntList(where_self_103, [0, 2, 3])
        sub_tensor_345 = torch.ops.aten.sub.Tensor(arg1374_1, arg1839_1);  arg1374_1 = arg1839_1 = None
        mul_tensor_983 = torch.ops.aten.mul.Tensor(where_self_103, sub_tensor_345)
        sum_dim_int_list_191 = torch.ops.aten.sum.dim_IntList(mul_tensor_983, [0, 2, 3]);  mul_tensor_983 = None
        mul_tensor_984 = torch.ops.aten.mul.Tensor(sum_dim_int_list_190, 0.0022675736961451248);  sum_dim_int_list_190 = None
        view_default_308 = torch.ops.aten.view.default(mul_tensor_984, [1, 336, 1, 1]);  mul_tensor_984 = None
        mul_tensor_985 = torch.ops.aten.mul.Tensor(sum_dim_int_list_191, 0.0022675736961451248)
        mul_tensor_986 = torch.ops.aten.mul.Tensor(arg1375_1, arg1375_1)
        mul_tensor_987 = torch.ops.aten.mul.Tensor(mul_tensor_985, mul_tensor_986);  mul_tensor_985 = mul_tensor_986 = None
        view_default_309 = torch.ops.aten.view.default(mul_tensor_987, [1, 336, 1, 1]);  mul_tensor_987 = None
        mul_tensor_988 = torch.ops.aten.mul.Tensor(arg1375_1, arg482_1);  arg482_1 = None
        view_default_310 = torch.ops.aten.view.default(mul_tensor_988, [1, 336, 1, 1]);  mul_tensor_988 = None
        mul_tensor_989 = torch.ops.aten.mul.Tensor(sub_tensor_345, view_default_309);  sub_tensor_345 = view_default_309 = None
        sub_tensor_346 = torch.ops.aten.sub.Tensor(where_self_103, mul_tensor_989);  where_self_103 = mul_tensor_989 = None
        sub_tensor_347 = torch.ops.aten.sub.Tensor(sub_tensor_346, view_default_308);  sub_tensor_346 = view_default_308 = None
        mul_tensor_990 = torch.ops.aten.mul.Tensor(sub_tensor_347, view_default_310);  sub_tensor_347 = view_default_310 = None
        mul_tensor_991 = torch.ops.aten.mul.Tensor(sum_dim_int_list_191, arg1375_1);  sum_dim_int_list_191 = arg1375_1 = None
        convolution_backward_default_191 = torch.ops.aten.convolution_backward.default(mul_tensor_990, arg1373_1, arg481_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_990 = arg1373_1 = arg481_1 = None
        getitem_573 = convolution_backward_default_191[0]
        getitem_574 = convolution_backward_default_191[1];  convolution_backward_default_191 = None
        convolution_backward_default_192 = torch.ops.aten.convolution_backward.default(getitem_573, relu_default_24, arg480_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_573 = relu_default_24 = arg480_1 = None
        getitem_576 = convolution_backward_default_192[0]
        getitem_577 = convolution_backward_default_192[1];  convolution_backward_default_192 = None
        new_zeros_default_104 = torch.ops.aten.new_zeros.default(getitem_576, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_104 = torch.ops.aten.where.self(le_scalar_67, new_zeros_default_104, getitem_576);  le_scalar_67 = new_zeros_default_104 = getitem_576 = None
        add_tensor_121 = torch.ops.aten.add.Tensor(add_tensor_120, where_self_104);  add_tensor_120 = where_self_104 = None
        sub_tensor_348 = torch.ops.aten.sub.Tensor(arg1371_1, arg1840_1);  arg1371_1 = arg1840_1 = None
        mul_tensor_992 = torch.ops.aten.mul.Tensor(slice_tensor_49, sub_tensor_348)
        sum_dim_int_list_192 = torch.ops.aten.sum.dim_IntList(mul_tensor_992, [0, 2, 3]);  mul_tensor_992 = None
        mul_tensor_993 = torch.ops.aten.mul.Tensor(sum_dim_int_list_192, 0.0022675736961451248)
        mul_tensor_994 = torch.ops.aten.mul.Tensor(arg1372_1, arg1372_1)
        mul_tensor_995 = torch.ops.aten.mul.Tensor(mul_tensor_993, mul_tensor_994);  mul_tensor_993 = mul_tensor_994 = None
        view_default_311 = torch.ops.aten.view.default(mul_tensor_995, [1, 336, 1, 1]);  mul_tensor_995 = None
        mul_tensor_996 = torch.ops.aten.mul.Tensor(arg1372_1, arg479_1);  arg479_1 = None
        view_default_312 = torch.ops.aten.view.default(mul_tensor_996, [1, 336, 1, 1]);  mul_tensor_996 = None
        mul_tensor_997 = torch.ops.aten.mul.Tensor(sub_tensor_348, view_default_311);  sub_tensor_348 = view_default_311 = None
        sub_tensor_349 = torch.ops.aten.sub.Tensor(slice_tensor_49, mul_tensor_997);  slice_tensor_49 = mul_tensor_997 = None
        sub_tensor_350 = torch.ops.aten.sub.Tensor(sub_tensor_349, view_default_305);  sub_tensor_349 = view_default_305 = None
        mul_tensor_998 = torch.ops.aten.mul.Tensor(sub_tensor_350, view_default_312);  sub_tensor_350 = view_default_312 = None
        mul_tensor_999 = torch.ops.aten.mul.Tensor(sum_dim_int_list_192, arg1372_1);  sum_dim_int_list_192 = arg1372_1 = None
        convolution_backward_default_193 = torch.ops.aten.convolution_backward.default(mul_tensor_998, arg1370_1, arg478_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_998 = arg1370_1 = arg478_1 = None
        getitem_579 = convolution_backward_default_193[0]
        getitem_580 = convolution_backward_default_193[1];  convolution_backward_default_193 = None
        convolution_backward_default_194 = torch.ops.aten.convolution_backward.default(getitem_579, arg1369_1, arg477_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_579 = arg477_1 = None
        getitem_582 = convolution_backward_default_194[0]
        getitem_583 = convolution_backward_default_194[1];  convolution_backward_default_194 = None
        le_scalar_70 = torch.ops.aten.le.Scalar(arg1369_1, 0);  arg1369_1 = None
        new_zeros_default_105 = torch.ops.aten.new_zeros.default(getitem_582, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_105 = torch.ops.aten.where.self(le_scalar_70, new_zeros_default_105, getitem_582);  le_scalar_70 = new_zeros_default_105 = getitem_582 = None
        sum_dim_int_list_193 = torch.ops.aten.sum.dim_IntList(where_self_105, [0, 2, 3])
        sub_tensor_351 = torch.ops.aten.sub.Tensor(arg1367_1, arg1841_1);  arg1367_1 = arg1841_1 = None
        mul_tensor_1000 = torch.ops.aten.mul.Tensor(where_self_105, sub_tensor_351)
        sum_dim_int_list_194 = torch.ops.aten.sum.dim_IntList(mul_tensor_1000, [0, 2, 3]);  mul_tensor_1000 = None
        mul_tensor_1001 = torch.ops.aten.mul.Tensor(sum_dim_int_list_193, 0.0022675736961451248);  sum_dim_int_list_193 = None
        view_default_313 = torch.ops.aten.view.default(mul_tensor_1001, [1, 336, 1, 1]);  mul_tensor_1001 = None
        mul_tensor_1002 = torch.ops.aten.mul.Tensor(sum_dim_int_list_194, 0.0022675736961451248)
        mul_tensor_1003 = torch.ops.aten.mul.Tensor(arg1368_1, arg1368_1)
        mul_tensor_1004 = torch.ops.aten.mul.Tensor(mul_tensor_1002, mul_tensor_1003);  mul_tensor_1002 = mul_tensor_1003 = None
        view_default_314 = torch.ops.aten.view.default(mul_tensor_1004, [1, 336, 1, 1]);  mul_tensor_1004 = None
        mul_tensor_1005 = torch.ops.aten.mul.Tensor(arg1368_1, arg476_1);  arg476_1 = None
        view_default_315 = torch.ops.aten.view.default(mul_tensor_1005, [1, 336, 1, 1]);  mul_tensor_1005 = None
        mul_tensor_1006 = torch.ops.aten.mul.Tensor(sub_tensor_351, view_default_314);  sub_tensor_351 = view_default_314 = None
        sub_tensor_352 = torch.ops.aten.sub.Tensor(where_self_105, mul_tensor_1006);  where_self_105 = mul_tensor_1006 = None
        sub_tensor_353 = torch.ops.aten.sub.Tensor(sub_tensor_352, view_default_313);  sub_tensor_352 = view_default_313 = None
        mul_tensor_1007 = torch.ops.aten.mul.Tensor(sub_tensor_353, view_default_315);  sub_tensor_353 = view_default_315 = None
        mul_tensor_1008 = torch.ops.aten.mul.Tensor(sum_dim_int_list_194, arg1368_1);  sum_dim_int_list_194 = arg1368_1 = None
        convolution_backward_default_195 = torch.ops.aten.convolution_backward.default(mul_tensor_1007, arg1366_1, arg475_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1007 = arg1366_1 = arg475_1 = None
        getitem_585 = convolution_backward_default_195[0]
        getitem_586 = convolution_backward_default_195[1];  convolution_backward_default_195 = None
        convolution_backward_default_196 = torch.ops.aten.convolution_backward.default(getitem_585, relu_default_23, arg474_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_585 = relu_default_23 = arg474_1 = None
        getitem_588 = convolution_backward_default_196[0]
        getitem_589 = convolution_backward_default_196[1];  convolution_backward_default_196 = None
        new_zeros_default_106 = torch.ops.aten.new_zeros.default(getitem_588, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_106 = torch.ops.aten.where.self(le_scalar_65, new_zeros_default_106, getitem_588);  le_scalar_65 = new_zeros_default_106 = getitem_588 = None
        add_tensor_122 = torch.ops.aten.add.Tensor(add_tensor_118, where_self_106);  add_tensor_118 = where_self_106 = None
        view_default_316 = torch.ops.aten.view.default(squeeze_dim_128, [1, 336, 1, 1]);  squeeze_dim_128 = None
        sum_dim_int_list_195 = torch.ops.aten.sum.dim_IntList(add_tensor_122, [0, 2, 3])
        sub_tensor_354 = torch.ops.aten.sub.Tensor(arg1363_1, view_default_316);  arg1363_1 = view_default_316 = None
        mul_tensor_1009 = torch.ops.aten.mul.Tensor(add_tensor_122, sub_tensor_354)
        sum_dim_int_list_196 = torch.ops.aten.sum.dim_IntList(mul_tensor_1009, [0, 2, 3]);  mul_tensor_1009 = None
        mul_tensor_1010 = torch.ops.aten.mul.Tensor(sum_dim_int_list_195, 0.0022675736961451248);  sum_dim_int_list_195 = None
        view_default_317 = torch.ops.aten.view.default(mul_tensor_1010, [1, 336, 1, 1]);  mul_tensor_1010 = None
        mul_tensor_1011 = torch.ops.aten.mul.Tensor(sum_dim_int_list_196, 0.0022675736961451248)
        mul_tensor_1012 = torch.ops.aten.mul.Tensor(squeeze_dim_131, squeeze_dim_131)
        mul_tensor_1013 = torch.ops.aten.mul.Tensor(mul_tensor_1011, mul_tensor_1012);  mul_tensor_1011 = mul_tensor_1012 = None
        view_default_318 = torch.ops.aten.view.default(mul_tensor_1013, [1, 336, 1, 1]);  mul_tensor_1013 = None
        mul_tensor_1014 = torch.ops.aten.mul.Tensor(squeeze_dim_131, arg472_1);  arg472_1 = None
        view_default_319 = torch.ops.aten.view.default(mul_tensor_1014, [1, 336, 1, 1]);  mul_tensor_1014 = None
        mul_tensor_1015 = torch.ops.aten.mul.Tensor(sub_tensor_354, view_default_318);  sub_tensor_354 = view_default_318 = None
        sub_tensor_355 = torch.ops.aten.sub.Tensor(add_tensor_122, mul_tensor_1015);  add_tensor_122 = mul_tensor_1015 = None
        sub_tensor_356 = torch.ops.aten.sub.Tensor(sub_tensor_355, view_default_317);  sub_tensor_355 = view_default_317 = None
        mul_tensor_1016 = torch.ops.aten.mul.Tensor(sub_tensor_356, view_default_319);  sub_tensor_356 = view_default_319 = None
        mul_tensor_1017 = torch.ops.aten.mul.Tensor(sum_dim_int_list_196, squeeze_dim_131);  sum_dim_int_list_196 = squeeze_dim_131 = None
        convolution_backward_default_197 = torch.ops.aten.convolution_backward.default(mul_tensor_1016, arg1362_1, arg471_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1016 = arg1362_1 = arg471_1 = None
        getitem_591 = convolution_backward_default_197[0]
        getitem_592 = convolution_backward_default_197[1];  convolution_backward_default_197 = None
        new_zeros_default_107 = torch.ops.aten.new_zeros.default(getitem_591, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_107 = torch.ops.aten.where.self(le_scalar_63, new_zeros_default_107, getitem_591);  le_scalar_63 = new_zeros_default_107 = getitem_591 = None
        add_tensor_123 = torch.ops.aten.add.Tensor(where_self_96, where_self_107);  where_self_96 = where_self_107 = None
        view_default_320 = torch.ops.aten.view.default(squeeze_dim_122, [1, 336, 1, 1]);  squeeze_dim_122 = None
        sum_dim_int_list_197 = torch.ops.aten.sum.dim_IntList(add_tensor_121, [0, 2, 3])
        sub_tensor_357 = torch.ops.aten.sub.Tensor(arg1359_1, view_default_320);  arg1359_1 = view_default_320 = None
        mul_tensor_1018 = torch.ops.aten.mul.Tensor(add_tensor_121, sub_tensor_357)
        sum_dim_int_list_198 = torch.ops.aten.sum.dim_IntList(mul_tensor_1018, [0, 2, 3]);  mul_tensor_1018 = None
        mul_tensor_1019 = torch.ops.aten.mul.Tensor(sum_dim_int_list_197, 0.0022675736961451248);  sum_dim_int_list_197 = None
        view_default_321 = torch.ops.aten.view.default(mul_tensor_1019, [1, 336, 1, 1]);  mul_tensor_1019 = None
        mul_tensor_1020 = torch.ops.aten.mul.Tensor(sum_dim_int_list_198, 0.0022675736961451248)
        mul_tensor_1021 = torch.ops.aten.mul.Tensor(squeeze_dim_125, squeeze_dim_125)
        mul_tensor_1022 = torch.ops.aten.mul.Tensor(mul_tensor_1020, mul_tensor_1021);  mul_tensor_1020 = mul_tensor_1021 = None
        view_default_322 = torch.ops.aten.view.default(mul_tensor_1022, [1, 336, 1, 1]);  mul_tensor_1022 = None
        mul_tensor_1023 = torch.ops.aten.mul.Tensor(squeeze_dim_125, arg469_1);  arg469_1 = None
        view_default_323 = torch.ops.aten.view.default(mul_tensor_1023, [1, 336, 1, 1]);  mul_tensor_1023 = None
        mul_tensor_1024 = torch.ops.aten.mul.Tensor(sub_tensor_357, view_default_322);  sub_tensor_357 = view_default_322 = None
        sub_tensor_358 = torch.ops.aten.sub.Tensor(add_tensor_121, mul_tensor_1024);  add_tensor_121 = mul_tensor_1024 = None
        sub_tensor_359 = torch.ops.aten.sub.Tensor(sub_tensor_358, view_default_321);  sub_tensor_358 = view_default_321 = None
        mul_tensor_1025 = torch.ops.aten.mul.Tensor(sub_tensor_359, view_default_323);  sub_tensor_359 = view_default_323 = None
        mul_tensor_1026 = torch.ops.aten.mul.Tensor(sum_dim_int_list_198, squeeze_dim_125);  sum_dim_int_list_198 = squeeze_dim_125 = None
        convolution_backward_default_198 = torch.ops.aten.convolution_backward.default(mul_tensor_1025, arg1320_1, arg468_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1025 = arg468_1 = None
        getitem_594 = convolution_backward_default_198[0]
        getitem_595 = convolution_backward_default_198[1];  convolution_backward_default_198 = None
        le_scalar_71 = torch.ops.aten.le.Scalar(arg1320_1, 0)
        new_zeros_default_108 = torch.ops.aten.new_zeros.default(getitem_594, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_108 = torch.ops.aten.where.self(le_scalar_71, new_zeros_default_108, getitem_594);  new_zeros_default_108 = getitem_594 = None
        slice_tensor_54 = torch.ops.aten.slice.Tensor(add_tensor_123, 1, 0, 336)
        slice_tensor_55 = torch.ops.aten.slice.Tensor(add_tensor_123, 1, 336, 672)
        slice_tensor_56 = torch.ops.aten.slice.Tensor(add_tensor_123, 1, 672, 1008)
        slice_tensor_57 = torch.ops.aten.slice.Tensor(add_tensor_123, 1, 1008, 1344)
        slice_tensor_58 = torch.ops.aten.slice.Tensor(add_tensor_123, 1, 1344, 1680)
        slice_tensor_59 = torch.ops.aten.slice.Tensor(add_tensor_123, 1, 1680, 2016);  add_tensor_123 = None
        sum_dim_int_list_199 = torch.ops.aten.sum.dim_IntList(slice_tensor_59, [0, 2, 3])
        sub_tensor_360 = torch.ops.aten.sub.Tensor(arg1357_1, arg1842_1);  arg1357_1 = arg1842_1 = None
        mul_tensor_1027 = torch.ops.aten.mul.Tensor(slice_tensor_59, sub_tensor_360)
        sum_dim_int_list_200 = torch.ops.aten.sum.dim_IntList(mul_tensor_1027, [0, 2, 3]);  mul_tensor_1027 = None
        mul_tensor_1028 = torch.ops.aten.mul.Tensor(sum_dim_int_list_199, 0.0022675736961451248);  sum_dim_int_list_199 = None
        view_default_324 = torch.ops.aten.view.default(mul_tensor_1028, [1, 336, 1, 1]);  mul_tensor_1028 = None
        mul_tensor_1029 = torch.ops.aten.mul.Tensor(sum_dim_int_list_200, 0.0022675736961451248)
        mul_tensor_1030 = torch.ops.aten.mul.Tensor(arg1358_1, arg1358_1)
        mul_tensor_1031 = torch.ops.aten.mul.Tensor(mul_tensor_1029, mul_tensor_1030);  mul_tensor_1029 = mul_tensor_1030 = None
        view_default_325 = torch.ops.aten.view.default(mul_tensor_1031, [1, 336, 1, 1]);  mul_tensor_1031 = None
        mul_tensor_1032 = torch.ops.aten.mul.Tensor(arg1358_1, arg467_1);  arg467_1 = None
        view_default_326 = torch.ops.aten.view.default(mul_tensor_1032, [1, 336, 1, 1]);  mul_tensor_1032 = None
        mul_tensor_1033 = torch.ops.aten.mul.Tensor(sub_tensor_360, view_default_325);  sub_tensor_360 = view_default_325 = None
        sub_tensor_361 = torch.ops.aten.sub.Tensor(slice_tensor_59, mul_tensor_1033);  mul_tensor_1033 = None
        sub_tensor_362 = torch.ops.aten.sub.Tensor(sub_tensor_361, view_default_324);  sub_tensor_361 = view_default_324 = None
        mul_tensor_1034 = torch.ops.aten.mul.Tensor(sub_tensor_362, view_default_326);  sub_tensor_362 = view_default_326 = None
        mul_tensor_1035 = torch.ops.aten.mul.Tensor(sum_dim_int_list_200, arg1358_1);  sum_dim_int_list_200 = arg1358_1 = None
        convolution_backward_default_199 = torch.ops.aten.convolution_backward.default(mul_tensor_1034, arg1356_1, arg466_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1034 = arg1356_1 = arg466_1 = None
        getitem_597 = convolution_backward_default_199[0]
        getitem_598 = convolution_backward_default_199[1];  convolution_backward_default_199 = None
        convolution_backward_default_200 = torch.ops.aten.convolution_backward.default(getitem_597, arg1355_1, arg465_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_597 = arg465_1 = None
        getitem_600 = convolution_backward_default_200[0]
        getitem_601 = convolution_backward_default_200[1];  convolution_backward_default_200 = None
        le_scalar_72 = torch.ops.aten.le.Scalar(arg1355_1, 0);  arg1355_1 = None
        new_zeros_default_109 = torch.ops.aten.new_zeros.default(getitem_600, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_109 = torch.ops.aten.where.self(le_scalar_72, new_zeros_default_109, getitem_600);  le_scalar_72 = new_zeros_default_109 = getitem_600 = None
        sum_dim_int_list_201 = torch.ops.aten.sum.dim_IntList(where_self_109, [0, 2, 3])
        sub_tensor_363 = torch.ops.aten.sub.Tensor(arg1353_1, arg1843_1);  arg1353_1 = arg1843_1 = None
        mul_tensor_1036 = torch.ops.aten.mul.Tensor(where_self_109, sub_tensor_363)
        sum_dim_int_list_202 = torch.ops.aten.sum.dim_IntList(mul_tensor_1036, [0, 2, 3]);  mul_tensor_1036 = None
        mul_tensor_1037 = torch.ops.aten.mul.Tensor(sum_dim_int_list_201, 0.0022675736961451248);  sum_dim_int_list_201 = None
        view_default_327 = torch.ops.aten.view.default(mul_tensor_1037, [1, 336, 1, 1]);  mul_tensor_1037 = None
        mul_tensor_1038 = torch.ops.aten.mul.Tensor(sum_dim_int_list_202, 0.0022675736961451248)
        mul_tensor_1039 = torch.ops.aten.mul.Tensor(arg1354_1, arg1354_1)
        mul_tensor_1040 = torch.ops.aten.mul.Tensor(mul_tensor_1038, mul_tensor_1039);  mul_tensor_1038 = mul_tensor_1039 = None
        view_default_328 = torch.ops.aten.view.default(mul_tensor_1040, [1, 336, 1, 1]);  mul_tensor_1040 = None
        mul_tensor_1041 = torch.ops.aten.mul.Tensor(arg1354_1, arg464_1);  arg464_1 = None
        view_default_329 = torch.ops.aten.view.default(mul_tensor_1041, [1, 336, 1, 1]);  mul_tensor_1041 = None
        mul_tensor_1042 = torch.ops.aten.mul.Tensor(sub_tensor_363, view_default_328);  sub_tensor_363 = view_default_328 = None
        sub_tensor_364 = torch.ops.aten.sub.Tensor(where_self_109, mul_tensor_1042);  where_self_109 = mul_tensor_1042 = None
        sub_tensor_365 = torch.ops.aten.sub.Tensor(sub_tensor_364, view_default_327);  sub_tensor_364 = view_default_327 = None
        mul_tensor_1043 = torch.ops.aten.mul.Tensor(sub_tensor_365, view_default_329);  sub_tensor_365 = view_default_329 = None
        mul_tensor_1044 = torch.ops.aten.mul.Tensor(sum_dim_int_list_202, arg1354_1);  sum_dim_int_list_202 = arg1354_1 = None
        convolution_backward_default_201 = torch.ops.aten.convolution_backward.default(mul_tensor_1043, arg1352_1, arg463_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1043 = arg1352_1 = arg463_1 = None
        getitem_603 = convolution_backward_default_201[0]
        getitem_604 = convolution_backward_default_201[1];  convolution_backward_default_201 = None
        convolution_backward_default_202 = torch.ops.aten.convolution_backward.default(getitem_603, relu_default_21, arg462_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_603 = arg462_1 = None
        getitem_606 = convolution_backward_default_202[0]
        getitem_607 = convolution_backward_default_202[1];  convolution_backward_default_202 = None
        le_scalar_73 = torch.ops.aten.le.Scalar(relu_default_21, 0)
        new_zeros_default_110 = torch.ops.aten.new_zeros.default(getitem_606, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_110 = torch.ops.aten.where.self(le_scalar_73, new_zeros_default_110, getitem_606);  new_zeros_default_110 = getitem_606 = None
        add_tensor_124 = torch.ops.aten.add.Tensor(slice_tensor_59, where_self_110);  slice_tensor_59 = where_self_110 = None
        avg_pool2d_backward_default_20 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_58, add_tensor_18, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_58 = add_tensor_18 = None
        add_tensor_125 = torch.ops.aten.add.Tensor(slice_tensor_54, avg_pool2d_backward_default_20);  slice_tensor_54 = None
        add_tensor_126 = torch.ops.aten.add.Tensor(add_tensor_125, avg_pool2d_backward_default_20);  add_tensor_125 = avg_pool2d_backward_default_20 = None
        add_tensor_127 = torch.ops.aten.add.Tensor(add_tensor_126, slice_tensor_57);  add_tensor_126 = None
        avg_pool2d_backward_default_21 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_57, add_tensor_19, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_57 = add_tensor_19 = None
        add_tensor_128 = torch.ops.aten.add.Tensor(add_tensor_124, avg_pool2d_backward_default_21);  add_tensor_124 = avg_pool2d_backward_default_21 = None
        sum_dim_int_list_203 = torch.ops.aten.sum.dim_IntList(slice_tensor_56, [0, 2, 3])
        sub_tensor_366 = torch.ops.aten.sub.Tensor(arg1350_1, arg1844_1);  arg1350_1 = arg1844_1 = None
        mul_tensor_1045 = torch.ops.aten.mul.Tensor(slice_tensor_56, sub_tensor_366)
        sum_dim_int_list_204 = torch.ops.aten.sum.dim_IntList(mul_tensor_1045, [0, 2, 3]);  mul_tensor_1045 = None
        mul_tensor_1046 = torch.ops.aten.mul.Tensor(sum_dim_int_list_203, 0.0022675736961451248);  sum_dim_int_list_203 = None
        view_default_330 = torch.ops.aten.view.default(mul_tensor_1046, [1, 336, 1, 1]);  mul_tensor_1046 = None
        mul_tensor_1047 = torch.ops.aten.mul.Tensor(sum_dim_int_list_204, 0.0022675736961451248)
        mul_tensor_1048 = torch.ops.aten.mul.Tensor(arg1351_1, arg1351_1)
        mul_tensor_1049 = torch.ops.aten.mul.Tensor(mul_tensor_1047, mul_tensor_1048);  mul_tensor_1047 = mul_tensor_1048 = None
        view_default_331 = torch.ops.aten.view.default(mul_tensor_1049, [1, 336, 1, 1]);  mul_tensor_1049 = None
        mul_tensor_1050 = torch.ops.aten.mul.Tensor(arg1351_1, arg461_1);  arg461_1 = None
        view_default_332 = torch.ops.aten.view.default(mul_tensor_1050, [1, 336, 1, 1]);  mul_tensor_1050 = None
        mul_tensor_1051 = torch.ops.aten.mul.Tensor(sub_tensor_366, view_default_331);  sub_tensor_366 = view_default_331 = None
        sub_tensor_367 = torch.ops.aten.sub.Tensor(slice_tensor_56, mul_tensor_1051);  mul_tensor_1051 = None
        sub_tensor_368 = torch.ops.aten.sub.Tensor(sub_tensor_367, view_default_330);  sub_tensor_367 = None
        mul_tensor_1052 = torch.ops.aten.mul.Tensor(sub_tensor_368, view_default_332);  sub_tensor_368 = view_default_332 = None
        mul_tensor_1053 = torch.ops.aten.mul.Tensor(sum_dim_int_list_204, arg1351_1);  sum_dim_int_list_204 = arg1351_1 = None
        convolution_backward_default_203 = torch.ops.aten.convolution_backward.default(mul_tensor_1052, arg1349_1, arg460_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1052 = arg1349_1 = arg460_1 = None
        getitem_609 = convolution_backward_default_203[0]
        getitem_610 = convolution_backward_default_203[1];  convolution_backward_default_203 = None
        convolution_backward_default_204 = torch.ops.aten.convolution_backward.default(getitem_609, arg1348_1, arg459_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_609 = arg459_1 = None
        getitem_612 = convolution_backward_default_204[0]
        getitem_613 = convolution_backward_default_204[1];  convolution_backward_default_204 = None
        le_scalar_74 = torch.ops.aten.le.Scalar(arg1348_1, 0);  arg1348_1 = None
        new_zeros_default_111 = torch.ops.aten.new_zeros.default(getitem_612, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_111 = torch.ops.aten.where.self(le_scalar_74, new_zeros_default_111, getitem_612);  le_scalar_74 = new_zeros_default_111 = getitem_612 = None
        sum_dim_int_list_205 = torch.ops.aten.sum.dim_IntList(where_self_111, [0, 2, 3])
        sub_tensor_369 = torch.ops.aten.sub.Tensor(arg1346_1, arg1845_1);  arg1346_1 = arg1845_1 = None
        mul_tensor_1054 = torch.ops.aten.mul.Tensor(where_self_111, sub_tensor_369)
        sum_dim_int_list_206 = torch.ops.aten.sum.dim_IntList(mul_tensor_1054, [0, 2, 3]);  mul_tensor_1054 = None
        mul_tensor_1055 = torch.ops.aten.mul.Tensor(sum_dim_int_list_205, 0.0022675736961451248);  sum_dim_int_list_205 = None
        view_default_333 = torch.ops.aten.view.default(mul_tensor_1055, [1, 336, 1, 1]);  mul_tensor_1055 = None
        mul_tensor_1056 = torch.ops.aten.mul.Tensor(sum_dim_int_list_206, 0.0022675736961451248)
        mul_tensor_1057 = torch.ops.aten.mul.Tensor(arg1347_1, arg1347_1)
        mul_tensor_1058 = torch.ops.aten.mul.Tensor(mul_tensor_1056, mul_tensor_1057);  mul_tensor_1056 = mul_tensor_1057 = None
        view_default_334 = torch.ops.aten.view.default(mul_tensor_1058, [1, 336, 1, 1]);  mul_tensor_1058 = None
        mul_tensor_1059 = torch.ops.aten.mul.Tensor(arg1347_1, arg458_1);  arg458_1 = None
        view_default_335 = torch.ops.aten.view.default(mul_tensor_1059, [1, 336, 1, 1]);  mul_tensor_1059 = None
        mul_tensor_1060 = torch.ops.aten.mul.Tensor(sub_tensor_369, view_default_334);  sub_tensor_369 = view_default_334 = None
        sub_tensor_370 = torch.ops.aten.sub.Tensor(where_self_111, mul_tensor_1060);  where_self_111 = mul_tensor_1060 = None
        sub_tensor_371 = torch.ops.aten.sub.Tensor(sub_tensor_370, view_default_333);  sub_tensor_370 = view_default_333 = None
        mul_tensor_1061 = torch.ops.aten.mul.Tensor(sub_tensor_371, view_default_335);  sub_tensor_371 = view_default_335 = None
        mul_tensor_1062 = torch.ops.aten.mul.Tensor(sum_dim_int_list_206, arg1347_1);  sum_dim_int_list_206 = arg1347_1 = None
        convolution_backward_default_205 = torch.ops.aten.convolution_backward.default(mul_tensor_1061, arg1345_1, arg457_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1061 = arg1345_1 = arg457_1 = None
        getitem_615 = convolution_backward_default_205[0]
        getitem_616 = convolution_backward_default_205[1];  convolution_backward_default_205 = None
        convolution_backward_default_206 = torch.ops.aten.convolution_backward.default(getitem_615, relu_default_22, arg456_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_615 = arg456_1 = None
        getitem_618 = convolution_backward_default_206[0]
        getitem_619 = convolution_backward_default_206[1];  convolution_backward_default_206 = None
        le_scalar_75 = torch.ops.aten.le.Scalar(relu_default_22, 0)
        new_zeros_default_112 = torch.ops.aten.new_zeros.default(getitem_618, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_112 = torch.ops.aten.where.self(le_scalar_75, new_zeros_default_112, getitem_618);  new_zeros_default_112 = getitem_618 = None
        add_tensor_129 = torch.ops.aten.add.Tensor(add_tensor_127, where_self_112);  add_tensor_127 = where_self_112 = None
        sub_tensor_372 = torch.ops.aten.sub.Tensor(arg1343_1, arg1846_1);  arg1343_1 = arg1846_1 = None
        mul_tensor_1063 = torch.ops.aten.mul.Tensor(slice_tensor_56, sub_tensor_372)
        sum_dim_int_list_207 = torch.ops.aten.sum.dim_IntList(mul_tensor_1063, [0, 2, 3]);  mul_tensor_1063 = None
        mul_tensor_1064 = torch.ops.aten.mul.Tensor(sum_dim_int_list_207, 0.0022675736961451248)
        mul_tensor_1065 = torch.ops.aten.mul.Tensor(arg1344_1, arg1344_1)
        mul_tensor_1066 = torch.ops.aten.mul.Tensor(mul_tensor_1064, mul_tensor_1065);  mul_tensor_1064 = mul_tensor_1065 = None
        view_default_336 = torch.ops.aten.view.default(mul_tensor_1066, [1, 336, 1, 1]);  mul_tensor_1066 = None
        mul_tensor_1067 = torch.ops.aten.mul.Tensor(arg1344_1, arg455_1);  arg455_1 = None
        view_default_337 = torch.ops.aten.view.default(mul_tensor_1067, [1, 336, 1, 1]);  mul_tensor_1067 = None
        mul_tensor_1068 = torch.ops.aten.mul.Tensor(sub_tensor_372, view_default_336);  sub_tensor_372 = view_default_336 = None
        sub_tensor_373 = torch.ops.aten.sub.Tensor(slice_tensor_56, mul_tensor_1068);  slice_tensor_56 = mul_tensor_1068 = None
        sub_tensor_374 = torch.ops.aten.sub.Tensor(sub_tensor_373, view_default_330);  sub_tensor_373 = view_default_330 = None
        mul_tensor_1069 = torch.ops.aten.mul.Tensor(sub_tensor_374, view_default_337);  sub_tensor_374 = view_default_337 = None
        mul_tensor_1070 = torch.ops.aten.mul.Tensor(sum_dim_int_list_207, arg1344_1);  sum_dim_int_list_207 = arg1344_1 = None
        convolution_backward_default_207 = torch.ops.aten.convolution_backward.default(mul_tensor_1069, arg1342_1, arg454_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1069 = arg1342_1 = arg454_1 = None
        getitem_621 = convolution_backward_default_207[0]
        getitem_622 = convolution_backward_default_207[1];  convolution_backward_default_207 = None
        convolution_backward_default_208 = torch.ops.aten.convolution_backward.default(getitem_621, arg1341_1, arg453_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_621 = arg453_1 = None
        getitem_624 = convolution_backward_default_208[0]
        getitem_625 = convolution_backward_default_208[1];  convolution_backward_default_208 = None
        le_scalar_76 = torch.ops.aten.le.Scalar(arg1341_1, 0);  arg1341_1 = None
        new_zeros_default_113 = torch.ops.aten.new_zeros.default(getitem_624, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_113 = torch.ops.aten.where.self(le_scalar_76, new_zeros_default_113, getitem_624);  le_scalar_76 = new_zeros_default_113 = getitem_624 = None
        sum_dim_int_list_208 = torch.ops.aten.sum.dim_IntList(where_self_113, [0, 2, 3])
        sub_tensor_375 = torch.ops.aten.sub.Tensor(arg1339_1, arg1847_1);  arg1339_1 = arg1847_1 = None
        mul_tensor_1071 = torch.ops.aten.mul.Tensor(where_self_113, sub_tensor_375)
        sum_dim_int_list_209 = torch.ops.aten.sum.dim_IntList(mul_tensor_1071, [0, 2, 3]);  mul_tensor_1071 = None
        mul_tensor_1072 = torch.ops.aten.mul.Tensor(sum_dim_int_list_208, 0.0022675736961451248);  sum_dim_int_list_208 = None
        view_default_338 = torch.ops.aten.view.default(mul_tensor_1072, [1, 336, 1, 1]);  mul_tensor_1072 = None
        mul_tensor_1073 = torch.ops.aten.mul.Tensor(sum_dim_int_list_209, 0.0022675736961451248)
        mul_tensor_1074 = torch.ops.aten.mul.Tensor(arg1340_1, arg1340_1)
        mul_tensor_1075 = torch.ops.aten.mul.Tensor(mul_tensor_1073, mul_tensor_1074);  mul_tensor_1073 = mul_tensor_1074 = None
        view_default_339 = torch.ops.aten.view.default(mul_tensor_1075, [1, 336, 1, 1]);  mul_tensor_1075 = None
        mul_tensor_1076 = torch.ops.aten.mul.Tensor(arg1340_1, arg452_1);  arg452_1 = None
        view_default_340 = torch.ops.aten.view.default(mul_tensor_1076, [1, 336, 1, 1]);  mul_tensor_1076 = None
        mul_tensor_1077 = torch.ops.aten.mul.Tensor(sub_tensor_375, view_default_339);  sub_tensor_375 = view_default_339 = None
        sub_tensor_376 = torch.ops.aten.sub.Tensor(where_self_113, mul_tensor_1077);  where_self_113 = mul_tensor_1077 = None
        sub_tensor_377 = torch.ops.aten.sub.Tensor(sub_tensor_376, view_default_338);  sub_tensor_376 = view_default_338 = None
        mul_tensor_1078 = torch.ops.aten.mul.Tensor(sub_tensor_377, view_default_340);  sub_tensor_377 = view_default_340 = None
        mul_tensor_1079 = torch.ops.aten.mul.Tensor(sum_dim_int_list_209, arg1340_1);  sum_dim_int_list_209 = arg1340_1 = None
        convolution_backward_default_209 = torch.ops.aten.convolution_backward.default(mul_tensor_1078, arg1338_1, arg451_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1078 = arg1338_1 = arg451_1 = None
        getitem_627 = convolution_backward_default_209[0]
        getitem_628 = convolution_backward_default_209[1];  convolution_backward_default_209 = None
        convolution_backward_default_210 = torch.ops.aten.convolution_backward.default(getitem_627, relu_default_22, arg450_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_627 = arg450_1 = None
        getitem_630 = convolution_backward_default_210[0]
        getitem_631 = convolution_backward_default_210[1];  convolution_backward_default_210 = None
        new_zeros_default_114 = torch.ops.aten.new_zeros.default(getitem_630, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_114 = torch.ops.aten.where.self(le_scalar_75, new_zeros_default_114, getitem_630);  new_zeros_default_114 = getitem_630 = None
        add_tensor_130 = torch.ops.aten.add.Tensor(add_tensor_129, where_self_114);  add_tensor_129 = where_self_114 = None
        sum_dim_int_list_210 = torch.ops.aten.sum.dim_IntList(slice_tensor_55, [0, 2, 3])
        sub_tensor_378 = torch.ops.aten.sub.Tensor(arg1336_1, arg1848_1);  arg1336_1 = arg1848_1 = None
        mul_tensor_1080 = torch.ops.aten.mul.Tensor(slice_tensor_55, sub_tensor_378)
        sum_dim_int_list_211 = torch.ops.aten.sum.dim_IntList(mul_tensor_1080, [0, 2, 3]);  mul_tensor_1080 = None
        mul_tensor_1081 = torch.ops.aten.mul.Tensor(sum_dim_int_list_210, 0.0022675736961451248);  sum_dim_int_list_210 = None
        view_default_341 = torch.ops.aten.view.default(mul_tensor_1081, [1, 336, 1, 1]);  mul_tensor_1081 = None
        mul_tensor_1082 = torch.ops.aten.mul.Tensor(sum_dim_int_list_211, 0.0022675736961451248)
        mul_tensor_1083 = torch.ops.aten.mul.Tensor(arg1337_1, arg1337_1)
        mul_tensor_1084 = torch.ops.aten.mul.Tensor(mul_tensor_1082, mul_tensor_1083);  mul_tensor_1082 = mul_tensor_1083 = None
        view_default_342 = torch.ops.aten.view.default(mul_tensor_1084, [1, 336, 1, 1]);  mul_tensor_1084 = None
        mul_tensor_1085 = torch.ops.aten.mul.Tensor(arg1337_1, arg449_1);  arg449_1 = None
        view_default_343 = torch.ops.aten.view.default(mul_tensor_1085, [1, 336, 1, 1]);  mul_tensor_1085 = None
        mul_tensor_1086 = torch.ops.aten.mul.Tensor(sub_tensor_378, view_default_342);  sub_tensor_378 = view_default_342 = None
        sub_tensor_379 = torch.ops.aten.sub.Tensor(slice_tensor_55, mul_tensor_1086);  mul_tensor_1086 = None
        sub_tensor_380 = torch.ops.aten.sub.Tensor(sub_tensor_379, view_default_341);  sub_tensor_379 = None
        mul_tensor_1087 = torch.ops.aten.mul.Tensor(sub_tensor_380, view_default_343);  sub_tensor_380 = view_default_343 = None
        mul_tensor_1088 = torch.ops.aten.mul.Tensor(sum_dim_int_list_211, arg1337_1);  sum_dim_int_list_211 = arg1337_1 = None
        convolution_backward_default_211 = torch.ops.aten.convolution_backward.default(mul_tensor_1087, arg1335_1, arg448_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1087 = arg1335_1 = arg448_1 = None
        getitem_633 = convolution_backward_default_211[0]
        getitem_634 = convolution_backward_default_211[1];  convolution_backward_default_211 = None
        convolution_backward_default_212 = torch.ops.aten.convolution_backward.default(getitem_633, arg1334_1, arg447_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_633 = arg447_1 = None
        getitem_636 = convolution_backward_default_212[0]
        getitem_637 = convolution_backward_default_212[1];  convolution_backward_default_212 = None
        le_scalar_77 = torch.ops.aten.le.Scalar(arg1334_1, 0);  arg1334_1 = None
        new_zeros_default_115 = torch.ops.aten.new_zeros.default(getitem_636, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_115 = torch.ops.aten.where.self(le_scalar_77, new_zeros_default_115, getitem_636);  le_scalar_77 = new_zeros_default_115 = getitem_636 = None
        sum_dim_int_list_212 = torch.ops.aten.sum.dim_IntList(where_self_115, [0, 2, 3])
        sub_tensor_381 = torch.ops.aten.sub.Tensor(arg1332_1, arg1849_1);  arg1332_1 = arg1849_1 = None
        mul_tensor_1089 = torch.ops.aten.mul.Tensor(where_self_115, sub_tensor_381)
        sum_dim_int_list_213 = torch.ops.aten.sum.dim_IntList(mul_tensor_1089, [0, 2, 3]);  mul_tensor_1089 = None
        mul_tensor_1090 = torch.ops.aten.mul.Tensor(sum_dim_int_list_212, 0.0022675736961451248);  sum_dim_int_list_212 = None
        view_default_344 = torch.ops.aten.view.default(mul_tensor_1090, [1, 336, 1, 1]);  mul_tensor_1090 = None
        mul_tensor_1091 = torch.ops.aten.mul.Tensor(sum_dim_int_list_213, 0.0022675736961451248)
        mul_tensor_1092 = torch.ops.aten.mul.Tensor(arg1333_1, arg1333_1)
        mul_tensor_1093 = torch.ops.aten.mul.Tensor(mul_tensor_1091, mul_tensor_1092);  mul_tensor_1091 = mul_tensor_1092 = None
        view_default_345 = torch.ops.aten.view.default(mul_tensor_1093, [1, 336, 1, 1]);  mul_tensor_1093 = None
        mul_tensor_1094 = torch.ops.aten.mul.Tensor(arg1333_1, arg446_1);  arg446_1 = None
        view_default_346 = torch.ops.aten.view.default(mul_tensor_1094, [1, 336, 1, 1]);  mul_tensor_1094 = None
        mul_tensor_1095 = torch.ops.aten.mul.Tensor(sub_tensor_381, view_default_345);  sub_tensor_381 = view_default_345 = None
        sub_tensor_382 = torch.ops.aten.sub.Tensor(where_self_115, mul_tensor_1095);  where_self_115 = mul_tensor_1095 = None
        sub_tensor_383 = torch.ops.aten.sub.Tensor(sub_tensor_382, view_default_344);  sub_tensor_382 = view_default_344 = None
        mul_tensor_1096 = torch.ops.aten.mul.Tensor(sub_tensor_383, view_default_346);  sub_tensor_383 = view_default_346 = None
        mul_tensor_1097 = torch.ops.aten.mul.Tensor(sum_dim_int_list_213, arg1333_1);  sum_dim_int_list_213 = arg1333_1 = None
        convolution_backward_default_213 = torch.ops.aten.convolution_backward.default(mul_tensor_1096, arg1331_1, arg445_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1096 = arg1331_1 = arg445_1 = None
        getitem_639 = convolution_backward_default_213[0]
        getitem_640 = convolution_backward_default_213[1];  convolution_backward_default_213 = None
        convolution_backward_default_214 = torch.ops.aten.convolution_backward.default(getitem_639, relu_default_22, arg444_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_639 = relu_default_22 = arg444_1 = None
        getitem_642 = convolution_backward_default_214[0]
        getitem_643 = convolution_backward_default_214[1];  convolution_backward_default_214 = None
        new_zeros_default_116 = torch.ops.aten.new_zeros.default(getitem_642, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_116 = torch.ops.aten.where.self(le_scalar_75, new_zeros_default_116, getitem_642);  le_scalar_75 = new_zeros_default_116 = getitem_642 = None
        add_tensor_131 = torch.ops.aten.add.Tensor(add_tensor_130, where_self_116);  add_tensor_130 = where_self_116 = None
        sub_tensor_384 = torch.ops.aten.sub.Tensor(arg1329_1, arg1850_1);  arg1329_1 = arg1850_1 = None
        mul_tensor_1098 = torch.ops.aten.mul.Tensor(slice_tensor_55, sub_tensor_384)
        sum_dim_int_list_214 = torch.ops.aten.sum.dim_IntList(mul_tensor_1098, [0, 2, 3]);  mul_tensor_1098 = None
        mul_tensor_1099 = torch.ops.aten.mul.Tensor(sum_dim_int_list_214, 0.0022675736961451248)
        mul_tensor_1100 = torch.ops.aten.mul.Tensor(arg1330_1, arg1330_1)
        mul_tensor_1101 = torch.ops.aten.mul.Tensor(mul_tensor_1099, mul_tensor_1100);  mul_tensor_1099 = mul_tensor_1100 = None
        view_default_347 = torch.ops.aten.view.default(mul_tensor_1101, [1, 336, 1, 1]);  mul_tensor_1101 = None
        mul_tensor_1102 = torch.ops.aten.mul.Tensor(arg1330_1, arg443_1);  arg443_1 = None
        view_default_348 = torch.ops.aten.view.default(mul_tensor_1102, [1, 336, 1, 1]);  mul_tensor_1102 = None
        mul_tensor_1103 = torch.ops.aten.mul.Tensor(sub_tensor_384, view_default_347);  sub_tensor_384 = view_default_347 = None
        sub_tensor_385 = torch.ops.aten.sub.Tensor(slice_tensor_55, mul_tensor_1103);  slice_tensor_55 = mul_tensor_1103 = None
        sub_tensor_386 = torch.ops.aten.sub.Tensor(sub_tensor_385, view_default_341);  sub_tensor_385 = view_default_341 = None
        mul_tensor_1104 = torch.ops.aten.mul.Tensor(sub_tensor_386, view_default_348);  sub_tensor_386 = view_default_348 = None
        mul_tensor_1105 = torch.ops.aten.mul.Tensor(sum_dim_int_list_214, arg1330_1);  sum_dim_int_list_214 = arg1330_1 = None
        convolution_backward_default_215 = torch.ops.aten.convolution_backward.default(mul_tensor_1104, arg1328_1, arg442_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1104 = arg1328_1 = arg442_1 = None
        getitem_645 = convolution_backward_default_215[0]
        getitem_646 = convolution_backward_default_215[1];  convolution_backward_default_215 = None
        convolution_backward_default_216 = torch.ops.aten.convolution_backward.default(getitem_645, arg1327_1, arg441_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_645 = arg441_1 = None
        getitem_648 = convolution_backward_default_216[0]
        getitem_649 = convolution_backward_default_216[1];  convolution_backward_default_216 = None
        le_scalar_78 = torch.ops.aten.le.Scalar(arg1327_1, 0);  arg1327_1 = None
        new_zeros_default_117 = torch.ops.aten.new_zeros.default(getitem_648, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_117 = torch.ops.aten.where.self(le_scalar_78, new_zeros_default_117, getitem_648);  le_scalar_78 = new_zeros_default_117 = getitem_648 = None
        sum_dim_int_list_215 = torch.ops.aten.sum.dim_IntList(where_self_117, [0, 2, 3])
        sub_tensor_387 = torch.ops.aten.sub.Tensor(arg1325_1, arg1851_1);  arg1325_1 = arg1851_1 = None
        mul_tensor_1106 = torch.ops.aten.mul.Tensor(where_self_117, sub_tensor_387)
        sum_dim_int_list_216 = torch.ops.aten.sum.dim_IntList(mul_tensor_1106, [0, 2, 3]);  mul_tensor_1106 = None
        mul_tensor_1107 = torch.ops.aten.mul.Tensor(sum_dim_int_list_215, 0.0022675736961451248);  sum_dim_int_list_215 = None
        view_default_349 = torch.ops.aten.view.default(mul_tensor_1107, [1, 336, 1, 1]);  mul_tensor_1107 = None
        mul_tensor_1108 = torch.ops.aten.mul.Tensor(sum_dim_int_list_216, 0.0022675736961451248)
        mul_tensor_1109 = torch.ops.aten.mul.Tensor(arg1326_1, arg1326_1)
        mul_tensor_1110 = torch.ops.aten.mul.Tensor(mul_tensor_1108, mul_tensor_1109);  mul_tensor_1108 = mul_tensor_1109 = None
        view_default_350 = torch.ops.aten.view.default(mul_tensor_1110, [1, 336, 1, 1]);  mul_tensor_1110 = None
        mul_tensor_1111 = torch.ops.aten.mul.Tensor(arg1326_1, arg440_1);  arg440_1 = None
        view_default_351 = torch.ops.aten.view.default(mul_tensor_1111, [1, 336, 1, 1]);  mul_tensor_1111 = None
        mul_tensor_1112 = torch.ops.aten.mul.Tensor(sub_tensor_387, view_default_350);  sub_tensor_387 = view_default_350 = None
        sub_tensor_388 = torch.ops.aten.sub.Tensor(where_self_117, mul_tensor_1112);  where_self_117 = mul_tensor_1112 = None
        sub_tensor_389 = torch.ops.aten.sub.Tensor(sub_tensor_388, view_default_349);  sub_tensor_388 = view_default_349 = None
        mul_tensor_1113 = torch.ops.aten.mul.Tensor(sub_tensor_389, view_default_351);  sub_tensor_389 = view_default_351 = None
        mul_tensor_1114 = torch.ops.aten.mul.Tensor(sum_dim_int_list_216, arg1326_1);  sum_dim_int_list_216 = arg1326_1 = None
        convolution_backward_default_217 = torch.ops.aten.convolution_backward.default(mul_tensor_1113, arg1324_1, arg439_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1113 = arg1324_1 = arg439_1 = None
        getitem_651 = convolution_backward_default_217[0]
        getitem_652 = convolution_backward_default_217[1];  convolution_backward_default_217 = None
        convolution_backward_default_218 = torch.ops.aten.convolution_backward.default(getitem_651, relu_default_21, arg438_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_651 = relu_default_21 = arg438_1 = None
        getitem_654 = convolution_backward_default_218[0]
        getitem_655 = convolution_backward_default_218[1];  convolution_backward_default_218 = None
        new_zeros_default_118 = torch.ops.aten.new_zeros.default(getitem_654, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_118 = torch.ops.aten.where.self(le_scalar_73, new_zeros_default_118, getitem_654);  le_scalar_73 = new_zeros_default_118 = getitem_654 = None
        add_tensor_132 = torch.ops.aten.add.Tensor(add_tensor_128, where_self_118);  add_tensor_128 = where_self_118 = None
        view_default_352 = torch.ops.aten.view.default(squeeze_dim_116, [1, 336, 1, 1]);  squeeze_dim_116 = None
        sum_dim_int_list_217 = torch.ops.aten.sum.dim_IntList(add_tensor_132, [0, 2, 3])
        sub_tensor_390 = torch.ops.aten.sub.Tensor(arg1321_1, view_default_352);  arg1321_1 = view_default_352 = None
        mul_tensor_1115 = torch.ops.aten.mul.Tensor(add_tensor_132, sub_tensor_390)
        sum_dim_int_list_218 = torch.ops.aten.sum.dim_IntList(mul_tensor_1115, [0, 2, 3]);  mul_tensor_1115 = None
        mul_tensor_1116 = torch.ops.aten.mul.Tensor(sum_dim_int_list_217, 0.0022675736961451248);  sum_dim_int_list_217 = None
        view_default_353 = torch.ops.aten.view.default(mul_tensor_1116, [1, 336, 1, 1]);  mul_tensor_1116 = None
        mul_tensor_1117 = torch.ops.aten.mul.Tensor(sum_dim_int_list_218, 0.0022675736961451248)
        mul_tensor_1118 = torch.ops.aten.mul.Tensor(squeeze_dim_119, squeeze_dim_119)
        mul_tensor_1119 = torch.ops.aten.mul.Tensor(mul_tensor_1117, mul_tensor_1118);  mul_tensor_1117 = mul_tensor_1118 = None
        view_default_354 = torch.ops.aten.view.default(mul_tensor_1119, [1, 336, 1, 1]);  mul_tensor_1119 = None
        mul_tensor_1120 = torch.ops.aten.mul.Tensor(squeeze_dim_119, arg436_1);  arg436_1 = None
        view_default_355 = torch.ops.aten.view.default(mul_tensor_1120, [1, 336, 1, 1]);  mul_tensor_1120 = None
        mul_tensor_1121 = torch.ops.aten.mul.Tensor(sub_tensor_390, view_default_354);  sub_tensor_390 = view_default_354 = None
        sub_tensor_391 = torch.ops.aten.sub.Tensor(add_tensor_132, mul_tensor_1121);  add_tensor_132 = mul_tensor_1121 = None
        sub_tensor_392 = torch.ops.aten.sub.Tensor(sub_tensor_391, view_default_353);  sub_tensor_391 = view_default_353 = None
        mul_tensor_1122 = torch.ops.aten.mul.Tensor(sub_tensor_392, view_default_355);  sub_tensor_392 = view_default_355 = None
        mul_tensor_1123 = torch.ops.aten.mul.Tensor(sum_dim_int_list_218, squeeze_dim_119);  sum_dim_int_list_218 = squeeze_dim_119 = None
        convolution_backward_default_219 = torch.ops.aten.convolution_backward.default(mul_tensor_1122, arg1320_1, arg435_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1122 = arg1320_1 = arg435_1 = None
        getitem_657 = convolution_backward_default_219[0]
        getitem_658 = convolution_backward_default_219[1];  convolution_backward_default_219 = None
        new_zeros_default_119 = torch.ops.aten.new_zeros.default(getitem_657, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_119 = torch.ops.aten.where.self(le_scalar_71, new_zeros_default_119, getitem_657);  le_scalar_71 = new_zeros_default_119 = getitem_657 = None
        add_tensor_133 = torch.ops.aten.add.Tensor(where_self_108, where_self_119);  where_self_108 = where_self_119 = None
        view_default_356 = torch.ops.aten.view.default(squeeze_dim_110, [1, 336, 1, 1]);  squeeze_dim_110 = None
        sum_dim_int_list_219 = torch.ops.aten.sum.dim_IntList(add_tensor_131, [0, 2, 3])
        sub_tensor_393 = torch.ops.aten.sub.Tensor(arg1317_1, view_default_356);  arg1317_1 = view_default_356 = None
        mul_tensor_1124 = torch.ops.aten.mul.Tensor(add_tensor_131, sub_tensor_393)
        sum_dim_int_list_220 = torch.ops.aten.sum.dim_IntList(mul_tensor_1124, [0, 2, 3]);  mul_tensor_1124 = None
        mul_tensor_1125 = torch.ops.aten.mul.Tensor(sum_dim_int_list_219, 0.0022675736961451248);  sum_dim_int_list_219 = None
        view_default_357 = torch.ops.aten.view.default(mul_tensor_1125, [1, 336, 1, 1]);  mul_tensor_1125 = None
        mul_tensor_1126 = torch.ops.aten.mul.Tensor(sum_dim_int_list_220, 0.0022675736961451248)
        mul_tensor_1127 = torch.ops.aten.mul.Tensor(squeeze_dim_113, squeeze_dim_113)
        mul_tensor_1128 = torch.ops.aten.mul.Tensor(mul_tensor_1126, mul_tensor_1127);  mul_tensor_1126 = mul_tensor_1127 = None
        view_default_358 = torch.ops.aten.view.default(mul_tensor_1128, [1, 336, 1, 1]);  mul_tensor_1128 = None
        mul_tensor_1129 = torch.ops.aten.mul.Tensor(squeeze_dim_113, arg433_1);  arg433_1 = None
        view_default_359 = torch.ops.aten.view.default(mul_tensor_1129, [1, 336, 1, 1]);  mul_tensor_1129 = None
        mul_tensor_1130 = torch.ops.aten.mul.Tensor(sub_tensor_393, view_default_358);  sub_tensor_393 = view_default_358 = None
        sub_tensor_394 = torch.ops.aten.sub.Tensor(add_tensor_131, mul_tensor_1130);  add_tensor_131 = mul_tensor_1130 = None
        sub_tensor_395 = torch.ops.aten.sub.Tensor(sub_tensor_394, view_default_357);  sub_tensor_394 = view_default_357 = None
        mul_tensor_1131 = torch.ops.aten.mul.Tensor(sub_tensor_395, view_default_359);  sub_tensor_395 = view_default_359 = None
        mul_tensor_1132 = torch.ops.aten.mul.Tensor(sum_dim_int_list_220, squeeze_dim_113);  sum_dim_int_list_220 = squeeze_dim_113 = None
        convolution_backward_default_220 = torch.ops.aten.convolution_backward.default(mul_tensor_1131, arg1278_1, arg432_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1131 = arg432_1 = None
        getitem_660 = convolution_backward_default_220[0]
        getitem_661 = convolution_backward_default_220[1];  convolution_backward_default_220 = None
        le_scalar_79 = torch.ops.aten.le.Scalar(arg1278_1, 0)
        new_zeros_default_120 = torch.ops.aten.new_zeros.default(getitem_660, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_120 = torch.ops.aten.where.self(le_scalar_79, new_zeros_default_120, getitem_660);  new_zeros_default_120 = getitem_660 = None
        slice_tensor_60 = torch.ops.aten.slice.Tensor(add_tensor_133, 1, 0, 336)
        slice_tensor_61 = torch.ops.aten.slice.Tensor(add_tensor_133, 1, 336, 672)
        slice_tensor_62 = torch.ops.aten.slice.Tensor(add_tensor_133, 1, 672, 1008)
        slice_tensor_63 = torch.ops.aten.slice.Tensor(add_tensor_133, 1, 1008, 1344)
        slice_tensor_64 = torch.ops.aten.slice.Tensor(add_tensor_133, 1, 1344, 1680)
        slice_tensor_65 = torch.ops.aten.slice.Tensor(add_tensor_133, 1, 1680, 2016);  add_tensor_133 = None
        sum_dim_int_list_221 = torch.ops.aten.sum.dim_IntList(slice_tensor_65, [0, 2, 3])
        sub_tensor_396 = torch.ops.aten.sub.Tensor(arg1315_1, arg1852_1);  arg1315_1 = arg1852_1 = None
        mul_tensor_1133 = torch.ops.aten.mul.Tensor(slice_tensor_65, sub_tensor_396)
        sum_dim_int_list_222 = torch.ops.aten.sum.dim_IntList(mul_tensor_1133, [0, 2, 3]);  mul_tensor_1133 = None
        mul_tensor_1134 = torch.ops.aten.mul.Tensor(sum_dim_int_list_221, 0.0022675736961451248);  sum_dim_int_list_221 = None
        view_default_360 = torch.ops.aten.view.default(mul_tensor_1134, [1, 336, 1, 1]);  mul_tensor_1134 = None
        mul_tensor_1135 = torch.ops.aten.mul.Tensor(sum_dim_int_list_222, 0.0022675736961451248)
        mul_tensor_1136 = torch.ops.aten.mul.Tensor(arg1316_1, arg1316_1)
        mul_tensor_1137 = torch.ops.aten.mul.Tensor(mul_tensor_1135, mul_tensor_1136);  mul_tensor_1135 = mul_tensor_1136 = None
        view_default_361 = torch.ops.aten.view.default(mul_tensor_1137, [1, 336, 1, 1]);  mul_tensor_1137 = None
        mul_tensor_1138 = torch.ops.aten.mul.Tensor(arg1316_1, arg431_1);  arg431_1 = None
        view_default_362 = torch.ops.aten.view.default(mul_tensor_1138, [1, 336, 1, 1]);  mul_tensor_1138 = None
        mul_tensor_1139 = torch.ops.aten.mul.Tensor(sub_tensor_396, view_default_361);  sub_tensor_396 = view_default_361 = None
        sub_tensor_397 = torch.ops.aten.sub.Tensor(slice_tensor_65, mul_tensor_1139);  mul_tensor_1139 = None
        sub_tensor_398 = torch.ops.aten.sub.Tensor(sub_tensor_397, view_default_360);  sub_tensor_397 = view_default_360 = None
        mul_tensor_1140 = torch.ops.aten.mul.Tensor(sub_tensor_398, view_default_362);  sub_tensor_398 = view_default_362 = None
        mul_tensor_1141 = torch.ops.aten.mul.Tensor(sum_dim_int_list_222, arg1316_1);  sum_dim_int_list_222 = arg1316_1 = None
        convolution_backward_default_221 = torch.ops.aten.convolution_backward.default(mul_tensor_1140, arg1314_1, arg430_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1140 = arg1314_1 = arg430_1 = None
        getitem_663 = convolution_backward_default_221[0]
        getitem_664 = convolution_backward_default_221[1];  convolution_backward_default_221 = None
        convolution_backward_default_222 = torch.ops.aten.convolution_backward.default(getitem_663, arg1313_1, arg429_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_663 = arg429_1 = None
        getitem_666 = convolution_backward_default_222[0]
        getitem_667 = convolution_backward_default_222[1];  convolution_backward_default_222 = None
        le_scalar_80 = torch.ops.aten.le.Scalar(arg1313_1, 0);  arg1313_1 = None
        new_zeros_default_121 = torch.ops.aten.new_zeros.default(getitem_666, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_121 = torch.ops.aten.where.self(le_scalar_80, new_zeros_default_121, getitem_666);  le_scalar_80 = new_zeros_default_121 = getitem_666 = None
        sum_dim_int_list_223 = torch.ops.aten.sum.dim_IntList(where_self_121, [0, 2, 3])
        sub_tensor_399 = torch.ops.aten.sub.Tensor(arg1311_1, arg1853_1);  arg1311_1 = arg1853_1 = None
        mul_tensor_1142 = torch.ops.aten.mul.Tensor(where_self_121, sub_tensor_399)
        sum_dim_int_list_224 = torch.ops.aten.sum.dim_IntList(mul_tensor_1142, [0, 2, 3]);  mul_tensor_1142 = None
        mul_tensor_1143 = torch.ops.aten.mul.Tensor(sum_dim_int_list_223, 0.0022675736961451248);  sum_dim_int_list_223 = None
        view_default_363 = torch.ops.aten.view.default(mul_tensor_1143, [1, 336, 1, 1]);  mul_tensor_1143 = None
        mul_tensor_1144 = torch.ops.aten.mul.Tensor(sum_dim_int_list_224, 0.0022675736961451248)
        mul_tensor_1145 = torch.ops.aten.mul.Tensor(arg1312_1, arg1312_1)
        mul_tensor_1146 = torch.ops.aten.mul.Tensor(mul_tensor_1144, mul_tensor_1145);  mul_tensor_1144 = mul_tensor_1145 = None
        view_default_364 = torch.ops.aten.view.default(mul_tensor_1146, [1, 336, 1, 1]);  mul_tensor_1146 = None
        mul_tensor_1147 = torch.ops.aten.mul.Tensor(arg1312_1, arg428_1);  arg428_1 = None
        view_default_365 = torch.ops.aten.view.default(mul_tensor_1147, [1, 336, 1, 1]);  mul_tensor_1147 = None
        mul_tensor_1148 = torch.ops.aten.mul.Tensor(sub_tensor_399, view_default_364);  sub_tensor_399 = view_default_364 = None
        sub_tensor_400 = torch.ops.aten.sub.Tensor(where_self_121, mul_tensor_1148);  where_self_121 = mul_tensor_1148 = None
        sub_tensor_401 = torch.ops.aten.sub.Tensor(sub_tensor_400, view_default_363);  sub_tensor_400 = view_default_363 = None
        mul_tensor_1149 = torch.ops.aten.mul.Tensor(sub_tensor_401, view_default_365);  sub_tensor_401 = view_default_365 = None
        mul_tensor_1150 = torch.ops.aten.mul.Tensor(sum_dim_int_list_224, arg1312_1);  sum_dim_int_list_224 = arg1312_1 = None
        convolution_backward_default_223 = torch.ops.aten.convolution_backward.default(mul_tensor_1149, arg1310_1, arg427_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1149 = arg1310_1 = arg427_1 = None
        getitem_669 = convolution_backward_default_223[0]
        getitem_670 = convolution_backward_default_223[1];  convolution_backward_default_223 = None
        convolution_backward_default_224 = torch.ops.aten.convolution_backward.default(getitem_669, relu_default_19, arg426_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_669 = arg426_1 = None
        getitem_672 = convolution_backward_default_224[0]
        getitem_673 = convolution_backward_default_224[1];  convolution_backward_default_224 = None
        le_scalar_81 = torch.ops.aten.le.Scalar(relu_default_19, 0)
        new_zeros_default_122 = torch.ops.aten.new_zeros.default(getitem_672, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_122 = torch.ops.aten.where.self(le_scalar_81, new_zeros_default_122, getitem_672);  new_zeros_default_122 = getitem_672 = None
        add_tensor_134 = torch.ops.aten.add.Tensor(slice_tensor_65, where_self_122);  slice_tensor_65 = where_self_122 = None
        avg_pool2d_backward_default_22 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_64, add_tensor_16, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_64 = add_tensor_16 = None
        add_tensor_135 = torch.ops.aten.add.Tensor(slice_tensor_60, avg_pool2d_backward_default_22);  slice_tensor_60 = None
        add_tensor_136 = torch.ops.aten.add.Tensor(add_tensor_135, avg_pool2d_backward_default_22);  add_tensor_135 = avg_pool2d_backward_default_22 = None
        add_tensor_137 = torch.ops.aten.add.Tensor(add_tensor_136, slice_tensor_63);  add_tensor_136 = None
        avg_pool2d_backward_default_23 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_63, add_tensor_17, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_63 = add_tensor_17 = None
        add_tensor_138 = torch.ops.aten.add.Tensor(add_tensor_134, avg_pool2d_backward_default_23);  add_tensor_134 = avg_pool2d_backward_default_23 = None
        sum_dim_int_list_225 = torch.ops.aten.sum.dim_IntList(slice_tensor_62, [0, 2, 3])
        sub_tensor_402 = torch.ops.aten.sub.Tensor(arg1308_1, arg1854_1);  arg1308_1 = arg1854_1 = None
        mul_tensor_1151 = torch.ops.aten.mul.Tensor(slice_tensor_62, sub_tensor_402)
        sum_dim_int_list_226 = torch.ops.aten.sum.dim_IntList(mul_tensor_1151, [0, 2, 3]);  mul_tensor_1151 = None
        mul_tensor_1152 = torch.ops.aten.mul.Tensor(sum_dim_int_list_225, 0.0022675736961451248);  sum_dim_int_list_225 = None
        view_default_366 = torch.ops.aten.view.default(mul_tensor_1152, [1, 336, 1, 1]);  mul_tensor_1152 = None
        mul_tensor_1153 = torch.ops.aten.mul.Tensor(sum_dim_int_list_226, 0.0022675736961451248)
        mul_tensor_1154 = torch.ops.aten.mul.Tensor(arg1309_1, arg1309_1)
        mul_tensor_1155 = torch.ops.aten.mul.Tensor(mul_tensor_1153, mul_tensor_1154);  mul_tensor_1153 = mul_tensor_1154 = None
        view_default_367 = torch.ops.aten.view.default(mul_tensor_1155, [1, 336, 1, 1]);  mul_tensor_1155 = None
        mul_tensor_1156 = torch.ops.aten.mul.Tensor(arg1309_1, arg425_1);  arg425_1 = None
        view_default_368 = torch.ops.aten.view.default(mul_tensor_1156, [1, 336, 1, 1]);  mul_tensor_1156 = None
        mul_tensor_1157 = torch.ops.aten.mul.Tensor(sub_tensor_402, view_default_367);  sub_tensor_402 = view_default_367 = None
        sub_tensor_403 = torch.ops.aten.sub.Tensor(slice_tensor_62, mul_tensor_1157);  mul_tensor_1157 = None
        sub_tensor_404 = torch.ops.aten.sub.Tensor(sub_tensor_403, view_default_366);  sub_tensor_403 = None
        mul_tensor_1158 = torch.ops.aten.mul.Tensor(sub_tensor_404, view_default_368);  sub_tensor_404 = view_default_368 = None
        mul_tensor_1159 = torch.ops.aten.mul.Tensor(sum_dim_int_list_226, arg1309_1);  sum_dim_int_list_226 = arg1309_1 = None
        convolution_backward_default_225 = torch.ops.aten.convolution_backward.default(mul_tensor_1158, arg1307_1, arg424_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1158 = arg1307_1 = arg424_1 = None
        getitem_675 = convolution_backward_default_225[0]
        getitem_676 = convolution_backward_default_225[1];  convolution_backward_default_225 = None
        convolution_backward_default_226 = torch.ops.aten.convolution_backward.default(getitem_675, arg1306_1, arg423_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_675 = arg423_1 = None
        getitem_678 = convolution_backward_default_226[0]
        getitem_679 = convolution_backward_default_226[1];  convolution_backward_default_226 = None
        le_scalar_82 = torch.ops.aten.le.Scalar(arg1306_1, 0);  arg1306_1 = None
        new_zeros_default_123 = torch.ops.aten.new_zeros.default(getitem_678, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_123 = torch.ops.aten.where.self(le_scalar_82, new_zeros_default_123, getitem_678);  le_scalar_82 = new_zeros_default_123 = getitem_678 = None
        sum_dim_int_list_227 = torch.ops.aten.sum.dim_IntList(where_self_123, [0, 2, 3])
        sub_tensor_405 = torch.ops.aten.sub.Tensor(arg1304_1, arg1855_1);  arg1304_1 = arg1855_1 = None
        mul_tensor_1160 = torch.ops.aten.mul.Tensor(where_self_123, sub_tensor_405)
        sum_dim_int_list_228 = torch.ops.aten.sum.dim_IntList(mul_tensor_1160, [0, 2, 3]);  mul_tensor_1160 = None
        mul_tensor_1161 = torch.ops.aten.mul.Tensor(sum_dim_int_list_227, 0.0022675736961451248);  sum_dim_int_list_227 = None
        view_default_369 = torch.ops.aten.view.default(mul_tensor_1161, [1, 336, 1, 1]);  mul_tensor_1161 = None
        mul_tensor_1162 = torch.ops.aten.mul.Tensor(sum_dim_int_list_228, 0.0022675736961451248)
        mul_tensor_1163 = torch.ops.aten.mul.Tensor(arg1305_1, arg1305_1)
        mul_tensor_1164 = torch.ops.aten.mul.Tensor(mul_tensor_1162, mul_tensor_1163);  mul_tensor_1162 = mul_tensor_1163 = None
        view_default_370 = torch.ops.aten.view.default(mul_tensor_1164, [1, 336, 1, 1]);  mul_tensor_1164 = None
        mul_tensor_1165 = torch.ops.aten.mul.Tensor(arg1305_1, arg422_1);  arg422_1 = None
        view_default_371 = torch.ops.aten.view.default(mul_tensor_1165, [1, 336, 1, 1]);  mul_tensor_1165 = None
        mul_tensor_1166 = torch.ops.aten.mul.Tensor(sub_tensor_405, view_default_370);  sub_tensor_405 = view_default_370 = None
        sub_tensor_406 = torch.ops.aten.sub.Tensor(where_self_123, mul_tensor_1166);  where_self_123 = mul_tensor_1166 = None
        sub_tensor_407 = torch.ops.aten.sub.Tensor(sub_tensor_406, view_default_369);  sub_tensor_406 = view_default_369 = None
        mul_tensor_1167 = torch.ops.aten.mul.Tensor(sub_tensor_407, view_default_371);  sub_tensor_407 = view_default_371 = None
        mul_tensor_1168 = torch.ops.aten.mul.Tensor(sum_dim_int_list_228, arg1305_1);  sum_dim_int_list_228 = arg1305_1 = None
        convolution_backward_default_227 = torch.ops.aten.convolution_backward.default(mul_tensor_1167, arg1303_1, arg421_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1167 = arg1303_1 = arg421_1 = None
        getitem_681 = convolution_backward_default_227[0]
        getitem_682 = convolution_backward_default_227[1];  convolution_backward_default_227 = None
        convolution_backward_default_228 = torch.ops.aten.convolution_backward.default(getitem_681, relu_default_20, arg420_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_681 = arg420_1 = None
        getitem_684 = convolution_backward_default_228[0]
        getitem_685 = convolution_backward_default_228[1];  convolution_backward_default_228 = None
        le_scalar_83 = torch.ops.aten.le.Scalar(relu_default_20, 0)
        new_zeros_default_124 = torch.ops.aten.new_zeros.default(getitem_684, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_124 = torch.ops.aten.where.self(le_scalar_83, new_zeros_default_124, getitem_684);  new_zeros_default_124 = getitem_684 = None
        add_tensor_139 = torch.ops.aten.add.Tensor(add_tensor_137, where_self_124);  add_tensor_137 = where_self_124 = None
        sub_tensor_408 = torch.ops.aten.sub.Tensor(arg1301_1, arg1856_1);  arg1301_1 = arg1856_1 = None
        mul_tensor_1169 = torch.ops.aten.mul.Tensor(slice_tensor_62, sub_tensor_408)
        sum_dim_int_list_229 = torch.ops.aten.sum.dim_IntList(mul_tensor_1169, [0, 2, 3]);  mul_tensor_1169 = None
        mul_tensor_1170 = torch.ops.aten.mul.Tensor(sum_dim_int_list_229, 0.0022675736961451248)
        mul_tensor_1171 = torch.ops.aten.mul.Tensor(arg1302_1, arg1302_1)
        mul_tensor_1172 = torch.ops.aten.mul.Tensor(mul_tensor_1170, mul_tensor_1171);  mul_tensor_1170 = mul_tensor_1171 = None
        view_default_372 = torch.ops.aten.view.default(mul_tensor_1172, [1, 336, 1, 1]);  mul_tensor_1172 = None
        mul_tensor_1173 = torch.ops.aten.mul.Tensor(arg1302_1, arg419_1);  arg419_1 = None
        view_default_373 = torch.ops.aten.view.default(mul_tensor_1173, [1, 336, 1, 1]);  mul_tensor_1173 = None
        mul_tensor_1174 = torch.ops.aten.mul.Tensor(sub_tensor_408, view_default_372);  sub_tensor_408 = view_default_372 = None
        sub_tensor_409 = torch.ops.aten.sub.Tensor(slice_tensor_62, mul_tensor_1174);  slice_tensor_62 = mul_tensor_1174 = None
        sub_tensor_410 = torch.ops.aten.sub.Tensor(sub_tensor_409, view_default_366);  sub_tensor_409 = view_default_366 = None
        mul_tensor_1175 = torch.ops.aten.mul.Tensor(sub_tensor_410, view_default_373);  sub_tensor_410 = view_default_373 = None
        mul_tensor_1176 = torch.ops.aten.mul.Tensor(sum_dim_int_list_229, arg1302_1);  sum_dim_int_list_229 = arg1302_1 = None
        convolution_backward_default_229 = torch.ops.aten.convolution_backward.default(mul_tensor_1175, arg1300_1, arg418_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1175 = arg1300_1 = arg418_1 = None
        getitem_687 = convolution_backward_default_229[0]
        getitem_688 = convolution_backward_default_229[1];  convolution_backward_default_229 = None
        convolution_backward_default_230 = torch.ops.aten.convolution_backward.default(getitem_687, arg1299_1, arg417_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_687 = arg417_1 = None
        getitem_690 = convolution_backward_default_230[0]
        getitem_691 = convolution_backward_default_230[1];  convolution_backward_default_230 = None
        le_scalar_84 = torch.ops.aten.le.Scalar(arg1299_1, 0);  arg1299_1 = None
        new_zeros_default_125 = torch.ops.aten.new_zeros.default(getitem_690, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_125 = torch.ops.aten.where.self(le_scalar_84, new_zeros_default_125, getitem_690);  le_scalar_84 = new_zeros_default_125 = getitem_690 = None
        sum_dim_int_list_230 = torch.ops.aten.sum.dim_IntList(where_self_125, [0, 2, 3])
        sub_tensor_411 = torch.ops.aten.sub.Tensor(arg1297_1, arg1857_1);  arg1297_1 = arg1857_1 = None
        mul_tensor_1177 = torch.ops.aten.mul.Tensor(where_self_125, sub_tensor_411)
        sum_dim_int_list_231 = torch.ops.aten.sum.dim_IntList(mul_tensor_1177, [0, 2, 3]);  mul_tensor_1177 = None
        mul_tensor_1178 = torch.ops.aten.mul.Tensor(sum_dim_int_list_230, 0.0022675736961451248);  sum_dim_int_list_230 = None
        view_default_374 = torch.ops.aten.view.default(mul_tensor_1178, [1, 336, 1, 1]);  mul_tensor_1178 = None
        mul_tensor_1179 = torch.ops.aten.mul.Tensor(sum_dim_int_list_231, 0.0022675736961451248)
        mul_tensor_1180 = torch.ops.aten.mul.Tensor(arg1298_1, arg1298_1)
        mul_tensor_1181 = torch.ops.aten.mul.Tensor(mul_tensor_1179, mul_tensor_1180);  mul_tensor_1179 = mul_tensor_1180 = None
        view_default_375 = torch.ops.aten.view.default(mul_tensor_1181, [1, 336, 1, 1]);  mul_tensor_1181 = None
        mul_tensor_1182 = torch.ops.aten.mul.Tensor(arg1298_1, arg416_1);  arg416_1 = None
        view_default_376 = torch.ops.aten.view.default(mul_tensor_1182, [1, 336, 1, 1]);  mul_tensor_1182 = None
        mul_tensor_1183 = torch.ops.aten.mul.Tensor(sub_tensor_411, view_default_375);  sub_tensor_411 = view_default_375 = None
        sub_tensor_412 = torch.ops.aten.sub.Tensor(where_self_125, mul_tensor_1183);  where_self_125 = mul_tensor_1183 = None
        sub_tensor_413 = torch.ops.aten.sub.Tensor(sub_tensor_412, view_default_374);  sub_tensor_412 = view_default_374 = None
        mul_tensor_1184 = torch.ops.aten.mul.Tensor(sub_tensor_413, view_default_376);  sub_tensor_413 = view_default_376 = None
        mul_tensor_1185 = torch.ops.aten.mul.Tensor(sum_dim_int_list_231, arg1298_1);  sum_dim_int_list_231 = arg1298_1 = None
        convolution_backward_default_231 = torch.ops.aten.convolution_backward.default(mul_tensor_1184, arg1296_1, arg415_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1184 = arg1296_1 = arg415_1 = None
        getitem_693 = convolution_backward_default_231[0]
        getitem_694 = convolution_backward_default_231[1];  convolution_backward_default_231 = None
        convolution_backward_default_232 = torch.ops.aten.convolution_backward.default(getitem_693, relu_default_20, arg414_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_693 = arg414_1 = None
        getitem_696 = convolution_backward_default_232[0]
        getitem_697 = convolution_backward_default_232[1];  convolution_backward_default_232 = None
        new_zeros_default_126 = torch.ops.aten.new_zeros.default(getitem_696, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_126 = torch.ops.aten.where.self(le_scalar_83, new_zeros_default_126, getitem_696);  new_zeros_default_126 = getitem_696 = None
        add_tensor_140 = torch.ops.aten.add.Tensor(add_tensor_139, where_self_126);  add_tensor_139 = where_self_126 = None
        sum_dim_int_list_232 = torch.ops.aten.sum.dim_IntList(slice_tensor_61, [0, 2, 3])
        sub_tensor_414 = torch.ops.aten.sub.Tensor(arg1294_1, arg1858_1);  arg1294_1 = arg1858_1 = None
        mul_tensor_1186 = torch.ops.aten.mul.Tensor(slice_tensor_61, sub_tensor_414)
        sum_dim_int_list_233 = torch.ops.aten.sum.dim_IntList(mul_tensor_1186, [0, 2, 3]);  mul_tensor_1186 = None
        mul_tensor_1187 = torch.ops.aten.mul.Tensor(sum_dim_int_list_232, 0.0022675736961451248);  sum_dim_int_list_232 = None
        view_default_377 = torch.ops.aten.view.default(mul_tensor_1187, [1, 336, 1, 1]);  mul_tensor_1187 = None
        mul_tensor_1188 = torch.ops.aten.mul.Tensor(sum_dim_int_list_233, 0.0022675736961451248)
        mul_tensor_1189 = torch.ops.aten.mul.Tensor(arg1295_1, arg1295_1)
        mul_tensor_1190 = torch.ops.aten.mul.Tensor(mul_tensor_1188, mul_tensor_1189);  mul_tensor_1188 = mul_tensor_1189 = None
        view_default_378 = torch.ops.aten.view.default(mul_tensor_1190, [1, 336, 1, 1]);  mul_tensor_1190 = None
        mul_tensor_1191 = torch.ops.aten.mul.Tensor(arg1295_1, arg413_1);  arg413_1 = None
        view_default_379 = torch.ops.aten.view.default(mul_tensor_1191, [1, 336, 1, 1]);  mul_tensor_1191 = None
        mul_tensor_1192 = torch.ops.aten.mul.Tensor(sub_tensor_414, view_default_378);  sub_tensor_414 = view_default_378 = None
        sub_tensor_415 = torch.ops.aten.sub.Tensor(slice_tensor_61, mul_tensor_1192);  mul_tensor_1192 = None
        sub_tensor_416 = torch.ops.aten.sub.Tensor(sub_tensor_415, view_default_377);  sub_tensor_415 = None
        mul_tensor_1193 = torch.ops.aten.mul.Tensor(sub_tensor_416, view_default_379);  sub_tensor_416 = view_default_379 = None
        mul_tensor_1194 = torch.ops.aten.mul.Tensor(sum_dim_int_list_233, arg1295_1);  sum_dim_int_list_233 = arg1295_1 = None
        convolution_backward_default_233 = torch.ops.aten.convolution_backward.default(mul_tensor_1193, arg1293_1, arg412_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1193 = arg1293_1 = arg412_1 = None
        getitem_699 = convolution_backward_default_233[0]
        getitem_700 = convolution_backward_default_233[1];  convolution_backward_default_233 = None
        convolution_backward_default_234 = torch.ops.aten.convolution_backward.default(getitem_699, arg1292_1, arg411_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_699 = arg411_1 = None
        getitem_702 = convolution_backward_default_234[0]
        getitem_703 = convolution_backward_default_234[1];  convolution_backward_default_234 = None
        le_scalar_85 = torch.ops.aten.le.Scalar(arg1292_1, 0);  arg1292_1 = None
        new_zeros_default_127 = torch.ops.aten.new_zeros.default(getitem_702, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_127 = torch.ops.aten.where.self(le_scalar_85, new_zeros_default_127, getitem_702);  le_scalar_85 = new_zeros_default_127 = getitem_702 = None
        sum_dim_int_list_234 = torch.ops.aten.sum.dim_IntList(where_self_127, [0, 2, 3])
        sub_tensor_417 = torch.ops.aten.sub.Tensor(arg1290_1, arg1859_1);  arg1290_1 = arg1859_1 = None
        mul_tensor_1195 = torch.ops.aten.mul.Tensor(where_self_127, sub_tensor_417)
        sum_dim_int_list_235 = torch.ops.aten.sum.dim_IntList(mul_tensor_1195, [0, 2, 3]);  mul_tensor_1195 = None
        mul_tensor_1196 = torch.ops.aten.mul.Tensor(sum_dim_int_list_234, 0.0022675736961451248);  sum_dim_int_list_234 = None
        view_default_380 = torch.ops.aten.view.default(mul_tensor_1196, [1, 336, 1, 1]);  mul_tensor_1196 = None
        mul_tensor_1197 = torch.ops.aten.mul.Tensor(sum_dim_int_list_235, 0.0022675736961451248)
        mul_tensor_1198 = torch.ops.aten.mul.Tensor(arg1291_1, arg1291_1)
        mul_tensor_1199 = torch.ops.aten.mul.Tensor(mul_tensor_1197, mul_tensor_1198);  mul_tensor_1197 = mul_tensor_1198 = None
        view_default_381 = torch.ops.aten.view.default(mul_tensor_1199, [1, 336, 1, 1]);  mul_tensor_1199 = None
        mul_tensor_1200 = torch.ops.aten.mul.Tensor(arg1291_1, arg410_1);  arg410_1 = None
        view_default_382 = torch.ops.aten.view.default(mul_tensor_1200, [1, 336, 1, 1]);  mul_tensor_1200 = None
        mul_tensor_1201 = torch.ops.aten.mul.Tensor(sub_tensor_417, view_default_381);  sub_tensor_417 = view_default_381 = None
        sub_tensor_418 = torch.ops.aten.sub.Tensor(where_self_127, mul_tensor_1201);  where_self_127 = mul_tensor_1201 = None
        sub_tensor_419 = torch.ops.aten.sub.Tensor(sub_tensor_418, view_default_380);  sub_tensor_418 = view_default_380 = None
        mul_tensor_1202 = torch.ops.aten.mul.Tensor(sub_tensor_419, view_default_382);  sub_tensor_419 = view_default_382 = None
        mul_tensor_1203 = torch.ops.aten.mul.Tensor(sum_dim_int_list_235, arg1291_1);  sum_dim_int_list_235 = arg1291_1 = None
        convolution_backward_default_235 = torch.ops.aten.convolution_backward.default(mul_tensor_1202, arg1289_1, arg409_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1202 = arg1289_1 = arg409_1 = None
        getitem_705 = convolution_backward_default_235[0]
        getitem_706 = convolution_backward_default_235[1];  convolution_backward_default_235 = None
        convolution_backward_default_236 = torch.ops.aten.convolution_backward.default(getitem_705, relu_default_20, arg408_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_705 = relu_default_20 = arg408_1 = None
        getitem_708 = convolution_backward_default_236[0]
        getitem_709 = convolution_backward_default_236[1];  convolution_backward_default_236 = None
        new_zeros_default_128 = torch.ops.aten.new_zeros.default(getitem_708, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_128 = torch.ops.aten.where.self(le_scalar_83, new_zeros_default_128, getitem_708);  le_scalar_83 = new_zeros_default_128 = getitem_708 = None
        add_tensor_141 = torch.ops.aten.add.Tensor(add_tensor_140, where_self_128);  add_tensor_140 = where_self_128 = None
        sub_tensor_420 = torch.ops.aten.sub.Tensor(arg1287_1, arg1860_1);  arg1287_1 = arg1860_1 = None
        mul_tensor_1204 = torch.ops.aten.mul.Tensor(slice_tensor_61, sub_tensor_420)
        sum_dim_int_list_236 = torch.ops.aten.sum.dim_IntList(mul_tensor_1204, [0, 2, 3]);  mul_tensor_1204 = None
        mul_tensor_1205 = torch.ops.aten.mul.Tensor(sum_dim_int_list_236, 0.0022675736961451248)
        mul_tensor_1206 = torch.ops.aten.mul.Tensor(arg1288_1, arg1288_1)
        mul_tensor_1207 = torch.ops.aten.mul.Tensor(mul_tensor_1205, mul_tensor_1206);  mul_tensor_1205 = mul_tensor_1206 = None
        view_default_383 = torch.ops.aten.view.default(mul_tensor_1207, [1, 336, 1, 1]);  mul_tensor_1207 = None
        mul_tensor_1208 = torch.ops.aten.mul.Tensor(arg1288_1, arg407_1);  arg407_1 = None
        view_default_384 = torch.ops.aten.view.default(mul_tensor_1208, [1, 336, 1, 1]);  mul_tensor_1208 = None
        mul_tensor_1209 = torch.ops.aten.mul.Tensor(sub_tensor_420, view_default_383);  sub_tensor_420 = view_default_383 = None
        sub_tensor_421 = torch.ops.aten.sub.Tensor(slice_tensor_61, mul_tensor_1209);  slice_tensor_61 = mul_tensor_1209 = None
        sub_tensor_422 = torch.ops.aten.sub.Tensor(sub_tensor_421, view_default_377);  sub_tensor_421 = view_default_377 = None
        mul_tensor_1210 = torch.ops.aten.mul.Tensor(sub_tensor_422, view_default_384);  sub_tensor_422 = view_default_384 = None
        mul_tensor_1211 = torch.ops.aten.mul.Tensor(sum_dim_int_list_236, arg1288_1);  sum_dim_int_list_236 = arg1288_1 = None
        convolution_backward_default_237 = torch.ops.aten.convolution_backward.default(mul_tensor_1210, arg1286_1, arg406_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1210 = arg1286_1 = arg406_1 = None
        getitem_711 = convolution_backward_default_237[0]
        getitem_712 = convolution_backward_default_237[1];  convolution_backward_default_237 = None
        convolution_backward_default_238 = torch.ops.aten.convolution_backward.default(getitem_711, arg1285_1, arg405_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_711 = arg405_1 = None
        getitem_714 = convolution_backward_default_238[0]
        getitem_715 = convolution_backward_default_238[1];  convolution_backward_default_238 = None
        le_scalar_86 = torch.ops.aten.le.Scalar(arg1285_1, 0);  arg1285_1 = None
        new_zeros_default_129 = torch.ops.aten.new_zeros.default(getitem_714, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_129 = torch.ops.aten.where.self(le_scalar_86, new_zeros_default_129, getitem_714);  le_scalar_86 = new_zeros_default_129 = getitem_714 = None
        sum_dim_int_list_237 = torch.ops.aten.sum.dim_IntList(where_self_129, [0, 2, 3])
        sub_tensor_423 = torch.ops.aten.sub.Tensor(arg1283_1, arg1861_1);  arg1283_1 = arg1861_1 = None
        mul_tensor_1212 = torch.ops.aten.mul.Tensor(where_self_129, sub_tensor_423)
        sum_dim_int_list_238 = torch.ops.aten.sum.dim_IntList(mul_tensor_1212, [0, 2, 3]);  mul_tensor_1212 = None
        mul_tensor_1213 = torch.ops.aten.mul.Tensor(sum_dim_int_list_237, 0.0022675736961451248);  sum_dim_int_list_237 = None
        view_default_385 = torch.ops.aten.view.default(mul_tensor_1213, [1, 336, 1, 1]);  mul_tensor_1213 = None
        mul_tensor_1214 = torch.ops.aten.mul.Tensor(sum_dim_int_list_238, 0.0022675736961451248)
        mul_tensor_1215 = torch.ops.aten.mul.Tensor(arg1284_1, arg1284_1)
        mul_tensor_1216 = torch.ops.aten.mul.Tensor(mul_tensor_1214, mul_tensor_1215);  mul_tensor_1214 = mul_tensor_1215 = None
        view_default_386 = torch.ops.aten.view.default(mul_tensor_1216, [1, 336, 1, 1]);  mul_tensor_1216 = None
        mul_tensor_1217 = torch.ops.aten.mul.Tensor(arg1284_1, arg404_1);  arg404_1 = None
        view_default_387 = torch.ops.aten.view.default(mul_tensor_1217, [1, 336, 1, 1]);  mul_tensor_1217 = None
        mul_tensor_1218 = torch.ops.aten.mul.Tensor(sub_tensor_423, view_default_386);  sub_tensor_423 = view_default_386 = None
        sub_tensor_424 = torch.ops.aten.sub.Tensor(where_self_129, mul_tensor_1218);  where_self_129 = mul_tensor_1218 = None
        sub_tensor_425 = torch.ops.aten.sub.Tensor(sub_tensor_424, view_default_385);  sub_tensor_424 = view_default_385 = None
        mul_tensor_1219 = torch.ops.aten.mul.Tensor(sub_tensor_425, view_default_387);  sub_tensor_425 = view_default_387 = None
        mul_tensor_1220 = torch.ops.aten.mul.Tensor(sum_dim_int_list_238, arg1284_1);  sum_dim_int_list_238 = arg1284_1 = None
        convolution_backward_default_239 = torch.ops.aten.convolution_backward.default(mul_tensor_1219, arg1282_1, arg403_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1219 = arg1282_1 = arg403_1 = None
        getitem_717 = convolution_backward_default_239[0]
        getitem_718 = convolution_backward_default_239[1];  convolution_backward_default_239 = None
        convolution_backward_default_240 = torch.ops.aten.convolution_backward.default(getitem_717, relu_default_19, arg402_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_717 = relu_default_19 = arg402_1 = None
        getitem_720 = convolution_backward_default_240[0]
        getitem_721 = convolution_backward_default_240[1];  convolution_backward_default_240 = None
        new_zeros_default_130 = torch.ops.aten.new_zeros.default(getitem_720, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_130 = torch.ops.aten.where.self(le_scalar_81, new_zeros_default_130, getitem_720);  le_scalar_81 = new_zeros_default_130 = getitem_720 = None
        add_tensor_142 = torch.ops.aten.add.Tensor(add_tensor_138, where_self_130);  add_tensor_138 = where_self_130 = None
        view_default_388 = torch.ops.aten.view.default(squeeze_dim_104, [1, 336, 1, 1]);  squeeze_dim_104 = None
        sum_dim_int_list_239 = torch.ops.aten.sum.dim_IntList(add_tensor_142, [0, 2, 3])
        sub_tensor_426 = torch.ops.aten.sub.Tensor(arg1279_1, view_default_388);  arg1279_1 = view_default_388 = None
        mul_tensor_1221 = torch.ops.aten.mul.Tensor(add_tensor_142, sub_tensor_426)
        sum_dim_int_list_240 = torch.ops.aten.sum.dim_IntList(mul_tensor_1221, [0, 2, 3]);  mul_tensor_1221 = None
        mul_tensor_1222 = torch.ops.aten.mul.Tensor(sum_dim_int_list_239, 0.0022675736961451248);  sum_dim_int_list_239 = None
        view_default_389 = torch.ops.aten.view.default(mul_tensor_1222, [1, 336, 1, 1]);  mul_tensor_1222 = None
        mul_tensor_1223 = torch.ops.aten.mul.Tensor(sum_dim_int_list_240, 0.0022675736961451248)
        mul_tensor_1224 = torch.ops.aten.mul.Tensor(squeeze_dim_107, squeeze_dim_107)
        mul_tensor_1225 = torch.ops.aten.mul.Tensor(mul_tensor_1223, mul_tensor_1224);  mul_tensor_1223 = mul_tensor_1224 = None
        view_default_390 = torch.ops.aten.view.default(mul_tensor_1225, [1, 336, 1, 1]);  mul_tensor_1225 = None
        mul_tensor_1226 = torch.ops.aten.mul.Tensor(squeeze_dim_107, arg400_1);  arg400_1 = None
        view_default_391 = torch.ops.aten.view.default(mul_tensor_1226, [1, 336, 1, 1]);  mul_tensor_1226 = None
        mul_tensor_1227 = torch.ops.aten.mul.Tensor(sub_tensor_426, view_default_390);  sub_tensor_426 = view_default_390 = None
        sub_tensor_427 = torch.ops.aten.sub.Tensor(add_tensor_142, mul_tensor_1227);  add_tensor_142 = mul_tensor_1227 = None
        sub_tensor_428 = torch.ops.aten.sub.Tensor(sub_tensor_427, view_default_389);  sub_tensor_427 = view_default_389 = None
        mul_tensor_1228 = torch.ops.aten.mul.Tensor(sub_tensor_428, view_default_391);  sub_tensor_428 = view_default_391 = None
        mul_tensor_1229 = torch.ops.aten.mul.Tensor(sum_dim_int_list_240, squeeze_dim_107);  sum_dim_int_list_240 = squeeze_dim_107 = None
        convolution_backward_default_241 = torch.ops.aten.convolution_backward.default(mul_tensor_1228, arg1278_1, arg399_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1228 = arg1278_1 = arg399_1 = None
        getitem_723 = convolution_backward_default_241[0]
        getitem_724 = convolution_backward_default_241[1];  convolution_backward_default_241 = None
        new_zeros_default_131 = torch.ops.aten.new_zeros.default(getitem_723, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_131 = torch.ops.aten.where.self(le_scalar_79, new_zeros_default_131, getitem_723);  le_scalar_79 = new_zeros_default_131 = getitem_723 = None
        add_tensor_143 = torch.ops.aten.add.Tensor(where_self_120, where_self_131);  where_self_120 = where_self_131 = None
        view_default_392 = torch.ops.aten.view.default(squeeze_dim_98, [1, 336, 1, 1]);  squeeze_dim_98 = None
        sum_dim_int_list_241 = torch.ops.aten.sum.dim_IntList(add_tensor_141, [0, 2, 3])
        sub_tensor_429 = torch.ops.aten.sub.Tensor(arg1275_1, view_default_392);  arg1275_1 = view_default_392 = None
        mul_tensor_1230 = torch.ops.aten.mul.Tensor(add_tensor_141, sub_tensor_429)
        sum_dim_int_list_242 = torch.ops.aten.sum.dim_IntList(mul_tensor_1230, [0, 2, 3]);  mul_tensor_1230 = None
        mul_tensor_1231 = torch.ops.aten.mul.Tensor(sum_dim_int_list_241, 0.0022675736961451248);  sum_dim_int_list_241 = None
        view_default_393 = torch.ops.aten.view.default(mul_tensor_1231, [1, 336, 1, 1]);  mul_tensor_1231 = None
        mul_tensor_1232 = torch.ops.aten.mul.Tensor(sum_dim_int_list_242, 0.0022675736961451248)
        mul_tensor_1233 = torch.ops.aten.mul.Tensor(squeeze_dim_101, squeeze_dim_101)
        mul_tensor_1234 = torch.ops.aten.mul.Tensor(mul_tensor_1232, mul_tensor_1233);  mul_tensor_1232 = mul_tensor_1233 = None
        view_default_394 = torch.ops.aten.view.default(mul_tensor_1234, [1, 336, 1, 1]);  mul_tensor_1234 = None
        mul_tensor_1235 = torch.ops.aten.mul.Tensor(squeeze_dim_101, arg397_1);  arg397_1 = None
        view_default_395 = torch.ops.aten.view.default(mul_tensor_1235, [1, 336, 1, 1]);  mul_tensor_1235 = None
        mul_tensor_1236 = torch.ops.aten.mul.Tensor(sub_tensor_429, view_default_394);  sub_tensor_429 = view_default_394 = None
        sub_tensor_430 = torch.ops.aten.sub.Tensor(add_tensor_141, mul_tensor_1236);  add_tensor_141 = mul_tensor_1236 = None
        sub_tensor_431 = torch.ops.aten.sub.Tensor(sub_tensor_430, view_default_393);  sub_tensor_430 = view_default_393 = None
        mul_tensor_1237 = torch.ops.aten.mul.Tensor(sub_tensor_431, view_default_395);  sub_tensor_431 = view_default_395 = None
        mul_tensor_1238 = torch.ops.aten.mul.Tensor(sum_dim_int_list_242, squeeze_dim_101);  sum_dim_int_list_242 = squeeze_dim_101 = None
        convolution_backward_default_242 = torch.ops.aten.convolution_backward.default(mul_tensor_1237, arg1236_1, arg396_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1237 = arg396_1 = None
        getitem_726 = convolution_backward_default_242[0]
        getitem_727 = convolution_backward_default_242[1];  convolution_backward_default_242 = None
        le_scalar_87 = torch.ops.aten.le.Scalar(arg1236_1, 0)
        new_zeros_default_132 = torch.ops.aten.new_zeros.default(getitem_726, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_132 = torch.ops.aten.where.self(le_scalar_87, new_zeros_default_132, getitem_726);  new_zeros_default_132 = getitem_726 = None
        slice_tensor_66 = torch.ops.aten.slice.Tensor(add_tensor_143, 1, 0, 336)
        slice_tensor_67 = torch.ops.aten.slice.Tensor(add_tensor_143, 1, 336, 672)
        slice_tensor_68 = torch.ops.aten.slice.Tensor(add_tensor_143, 1, 672, 1008)
        slice_tensor_69 = torch.ops.aten.slice.Tensor(add_tensor_143, 1, 1008, 1344)
        slice_tensor_70 = torch.ops.aten.slice.Tensor(add_tensor_143, 1, 1344, 1680)
        slice_tensor_71 = torch.ops.aten.slice.Tensor(add_tensor_143, 1, 1680, 2016);  add_tensor_143 = None
        sum_dim_int_list_243 = torch.ops.aten.sum.dim_IntList(slice_tensor_71, [0, 2, 3])
        sub_tensor_432 = torch.ops.aten.sub.Tensor(arg1273_1, arg1862_1);  arg1273_1 = arg1862_1 = None
        mul_tensor_1239 = torch.ops.aten.mul.Tensor(slice_tensor_71, sub_tensor_432)
        sum_dim_int_list_244 = torch.ops.aten.sum.dim_IntList(mul_tensor_1239, [0, 2, 3]);  mul_tensor_1239 = None
        mul_tensor_1240 = torch.ops.aten.mul.Tensor(sum_dim_int_list_243, 0.0022675736961451248);  sum_dim_int_list_243 = None
        view_default_396 = torch.ops.aten.view.default(mul_tensor_1240, [1, 336, 1, 1]);  mul_tensor_1240 = None
        mul_tensor_1241 = torch.ops.aten.mul.Tensor(sum_dim_int_list_244, 0.0022675736961451248)
        mul_tensor_1242 = torch.ops.aten.mul.Tensor(arg1274_1, arg1274_1)
        mul_tensor_1243 = torch.ops.aten.mul.Tensor(mul_tensor_1241, mul_tensor_1242);  mul_tensor_1241 = mul_tensor_1242 = None
        view_default_397 = torch.ops.aten.view.default(mul_tensor_1243, [1, 336, 1, 1]);  mul_tensor_1243 = None
        mul_tensor_1244 = torch.ops.aten.mul.Tensor(arg1274_1, arg395_1);  arg395_1 = None
        view_default_398 = torch.ops.aten.view.default(mul_tensor_1244, [1, 336, 1, 1]);  mul_tensor_1244 = None
        mul_tensor_1245 = torch.ops.aten.mul.Tensor(sub_tensor_432, view_default_397);  sub_tensor_432 = view_default_397 = None
        sub_tensor_433 = torch.ops.aten.sub.Tensor(slice_tensor_71, mul_tensor_1245);  mul_tensor_1245 = None
        sub_tensor_434 = torch.ops.aten.sub.Tensor(sub_tensor_433, view_default_396);  sub_tensor_433 = view_default_396 = None
        mul_tensor_1246 = torch.ops.aten.mul.Tensor(sub_tensor_434, view_default_398);  sub_tensor_434 = view_default_398 = None
        mul_tensor_1247 = torch.ops.aten.mul.Tensor(sum_dim_int_list_244, arg1274_1);  sum_dim_int_list_244 = arg1274_1 = None
        convolution_backward_default_243 = torch.ops.aten.convolution_backward.default(mul_tensor_1246, arg1272_1, arg394_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1246 = arg1272_1 = arg394_1 = None
        getitem_729 = convolution_backward_default_243[0]
        getitem_730 = convolution_backward_default_243[1];  convolution_backward_default_243 = None
        convolution_backward_default_244 = torch.ops.aten.convolution_backward.default(getitem_729, arg1271_1, arg393_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_729 = arg393_1 = None
        getitem_732 = convolution_backward_default_244[0]
        getitem_733 = convolution_backward_default_244[1];  convolution_backward_default_244 = None
        le_scalar_88 = torch.ops.aten.le.Scalar(arg1271_1, 0);  arg1271_1 = None
        new_zeros_default_133 = torch.ops.aten.new_zeros.default(getitem_732, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_133 = torch.ops.aten.where.self(le_scalar_88, new_zeros_default_133, getitem_732);  le_scalar_88 = new_zeros_default_133 = getitem_732 = None
        sum_dim_int_list_245 = torch.ops.aten.sum.dim_IntList(where_self_133, [0, 2, 3])
        sub_tensor_435 = torch.ops.aten.sub.Tensor(arg1269_1, arg1863_1);  arg1269_1 = arg1863_1 = None
        mul_tensor_1248 = torch.ops.aten.mul.Tensor(where_self_133, sub_tensor_435)
        sum_dim_int_list_246 = torch.ops.aten.sum.dim_IntList(mul_tensor_1248, [0, 2, 3]);  mul_tensor_1248 = None
        mul_tensor_1249 = torch.ops.aten.mul.Tensor(sum_dim_int_list_245, 0.0022675736961451248);  sum_dim_int_list_245 = None
        view_default_399 = torch.ops.aten.view.default(mul_tensor_1249, [1, 336, 1, 1]);  mul_tensor_1249 = None
        mul_tensor_1250 = torch.ops.aten.mul.Tensor(sum_dim_int_list_246, 0.0022675736961451248)
        mul_tensor_1251 = torch.ops.aten.mul.Tensor(arg1270_1, arg1270_1)
        mul_tensor_1252 = torch.ops.aten.mul.Tensor(mul_tensor_1250, mul_tensor_1251);  mul_tensor_1250 = mul_tensor_1251 = None
        view_default_400 = torch.ops.aten.view.default(mul_tensor_1252, [1, 336, 1, 1]);  mul_tensor_1252 = None
        mul_tensor_1253 = torch.ops.aten.mul.Tensor(arg1270_1, arg392_1);  arg392_1 = None
        view_default_401 = torch.ops.aten.view.default(mul_tensor_1253, [1, 336, 1, 1]);  mul_tensor_1253 = None
        mul_tensor_1254 = torch.ops.aten.mul.Tensor(sub_tensor_435, view_default_400);  sub_tensor_435 = view_default_400 = None
        sub_tensor_436 = torch.ops.aten.sub.Tensor(where_self_133, mul_tensor_1254);  where_self_133 = mul_tensor_1254 = None
        sub_tensor_437 = torch.ops.aten.sub.Tensor(sub_tensor_436, view_default_399);  sub_tensor_436 = view_default_399 = None
        mul_tensor_1255 = torch.ops.aten.mul.Tensor(sub_tensor_437, view_default_401);  sub_tensor_437 = view_default_401 = None
        mul_tensor_1256 = torch.ops.aten.mul.Tensor(sum_dim_int_list_246, arg1270_1);  sum_dim_int_list_246 = arg1270_1 = None
        convolution_backward_default_245 = torch.ops.aten.convolution_backward.default(mul_tensor_1255, arg1268_1, arg391_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1255 = arg1268_1 = arg391_1 = None
        getitem_735 = convolution_backward_default_245[0]
        getitem_736 = convolution_backward_default_245[1];  convolution_backward_default_245 = None
        convolution_backward_default_246 = torch.ops.aten.convolution_backward.default(getitem_735, relu_default_17, arg390_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_735 = arg390_1 = None
        getitem_738 = convolution_backward_default_246[0]
        getitem_739 = convolution_backward_default_246[1];  convolution_backward_default_246 = None
        le_scalar_89 = torch.ops.aten.le.Scalar(relu_default_17, 0)
        new_zeros_default_134 = torch.ops.aten.new_zeros.default(getitem_738, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_134 = torch.ops.aten.where.self(le_scalar_89, new_zeros_default_134, getitem_738);  new_zeros_default_134 = getitem_738 = None
        add_tensor_144 = torch.ops.aten.add.Tensor(slice_tensor_71, where_self_134);  slice_tensor_71 = where_self_134 = None
        avg_pool2d_backward_default_24 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_70, add_tensor_14, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_70 = add_tensor_14 = None
        add_tensor_145 = torch.ops.aten.add.Tensor(slice_tensor_66, avg_pool2d_backward_default_24);  slice_tensor_66 = None
        add_tensor_146 = torch.ops.aten.add.Tensor(add_tensor_145, avg_pool2d_backward_default_24);  add_tensor_145 = avg_pool2d_backward_default_24 = None
        add_tensor_147 = torch.ops.aten.add.Tensor(add_tensor_146, slice_tensor_69);  add_tensor_146 = None
        avg_pool2d_backward_default_25 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_69, add_tensor_15, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_69 = add_tensor_15 = None
        add_tensor_148 = torch.ops.aten.add.Tensor(add_tensor_144, avg_pool2d_backward_default_25);  add_tensor_144 = avg_pool2d_backward_default_25 = None
        sum_dim_int_list_247 = torch.ops.aten.sum.dim_IntList(slice_tensor_68, [0, 2, 3])
        sub_tensor_438 = torch.ops.aten.sub.Tensor(arg1266_1, arg1864_1);  arg1266_1 = arg1864_1 = None
        mul_tensor_1257 = torch.ops.aten.mul.Tensor(slice_tensor_68, sub_tensor_438)
        sum_dim_int_list_248 = torch.ops.aten.sum.dim_IntList(mul_tensor_1257, [0, 2, 3]);  mul_tensor_1257 = None
        mul_tensor_1258 = torch.ops.aten.mul.Tensor(sum_dim_int_list_247, 0.0022675736961451248);  sum_dim_int_list_247 = None
        view_default_402 = torch.ops.aten.view.default(mul_tensor_1258, [1, 336, 1, 1]);  mul_tensor_1258 = None
        mul_tensor_1259 = torch.ops.aten.mul.Tensor(sum_dim_int_list_248, 0.0022675736961451248)
        mul_tensor_1260 = torch.ops.aten.mul.Tensor(arg1267_1, arg1267_1)
        mul_tensor_1261 = torch.ops.aten.mul.Tensor(mul_tensor_1259, mul_tensor_1260);  mul_tensor_1259 = mul_tensor_1260 = None
        view_default_403 = torch.ops.aten.view.default(mul_tensor_1261, [1, 336, 1, 1]);  mul_tensor_1261 = None
        mul_tensor_1262 = torch.ops.aten.mul.Tensor(arg1267_1, arg389_1);  arg389_1 = None
        view_default_404 = torch.ops.aten.view.default(mul_tensor_1262, [1, 336, 1, 1]);  mul_tensor_1262 = None
        mul_tensor_1263 = torch.ops.aten.mul.Tensor(sub_tensor_438, view_default_403);  sub_tensor_438 = view_default_403 = None
        sub_tensor_439 = torch.ops.aten.sub.Tensor(slice_tensor_68, mul_tensor_1263);  mul_tensor_1263 = None
        sub_tensor_440 = torch.ops.aten.sub.Tensor(sub_tensor_439, view_default_402);  sub_tensor_439 = None
        mul_tensor_1264 = torch.ops.aten.mul.Tensor(sub_tensor_440, view_default_404);  sub_tensor_440 = view_default_404 = None
        mul_tensor_1265 = torch.ops.aten.mul.Tensor(sum_dim_int_list_248, arg1267_1);  sum_dim_int_list_248 = arg1267_1 = None
        convolution_backward_default_247 = torch.ops.aten.convolution_backward.default(mul_tensor_1264, arg1265_1, arg388_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1264 = arg1265_1 = arg388_1 = None
        getitem_741 = convolution_backward_default_247[0]
        getitem_742 = convolution_backward_default_247[1];  convolution_backward_default_247 = None
        convolution_backward_default_248 = torch.ops.aten.convolution_backward.default(getitem_741, arg1264_1, arg387_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_741 = arg387_1 = None
        getitem_744 = convolution_backward_default_248[0]
        getitem_745 = convolution_backward_default_248[1];  convolution_backward_default_248 = None
        le_scalar_90 = torch.ops.aten.le.Scalar(arg1264_1, 0);  arg1264_1 = None
        new_zeros_default_135 = torch.ops.aten.new_zeros.default(getitem_744, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_135 = torch.ops.aten.where.self(le_scalar_90, new_zeros_default_135, getitem_744);  le_scalar_90 = new_zeros_default_135 = getitem_744 = None
        sum_dim_int_list_249 = torch.ops.aten.sum.dim_IntList(where_self_135, [0, 2, 3])
        sub_tensor_441 = torch.ops.aten.sub.Tensor(arg1262_1, arg1865_1);  arg1262_1 = arg1865_1 = None
        mul_tensor_1266 = torch.ops.aten.mul.Tensor(where_self_135, sub_tensor_441)
        sum_dim_int_list_250 = torch.ops.aten.sum.dim_IntList(mul_tensor_1266, [0, 2, 3]);  mul_tensor_1266 = None
        mul_tensor_1267 = torch.ops.aten.mul.Tensor(sum_dim_int_list_249, 0.0022675736961451248);  sum_dim_int_list_249 = None
        view_default_405 = torch.ops.aten.view.default(mul_tensor_1267, [1, 336, 1, 1]);  mul_tensor_1267 = None
        mul_tensor_1268 = torch.ops.aten.mul.Tensor(sum_dim_int_list_250, 0.0022675736961451248)
        mul_tensor_1269 = torch.ops.aten.mul.Tensor(arg1263_1, arg1263_1)
        mul_tensor_1270 = torch.ops.aten.mul.Tensor(mul_tensor_1268, mul_tensor_1269);  mul_tensor_1268 = mul_tensor_1269 = None
        view_default_406 = torch.ops.aten.view.default(mul_tensor_1270, [1, 336, 1, 1]);  mul_tensor_1270 = None
        mul_tensor_1271 = torch.ops.aten.mul.Tensor(arg1263_1, arg386_1);  arg386_1 = None
        view_default_407 = torch.ops.aten.view.default(mul_tensor_1271, [1, 336, 1, 1]);  mul_tensor_1271 = None
        mul_tensor_1272 = torch.ops.aten.mul.Tensor(sub_tensor_441, view_default_406);  sub_tensor_441 = view_default_406 = None
        sub_tensor_442 = torch.ops.aten.sub.Tensor(where_self_135, mul_tensor_1272);  where_self_135 = mul_tensor_1272 = None
        sub_tensor_443 = torch.ops.aten.sub.Tensor(sub_tensor_442, view_default_405);  sub_tensor_442 = view_default_405 = None
        mul_tensor_1273 = torch.ops.aten.mul.Tensor(sub_tensor_443, view_default_407);  sub_tensor_443 = view_default_407 = None
        mul_tensor_1274 = torch.ops.aten.mul.Tensor(sum_dim_int_list_250, arg1263_1);  sum_dim_int_list_250 = arg1263_1 = None
        convolution_backward_default_249 = torch.ops.aten.convolution_backward.default(mul_tensor_1273, arg1261_1, arg385_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1273 = arg1261_1 = arg385_1 = None
        getitem_747 = convolution_backward_default_249[0]
        getitem_748 = convolution_backward_default_249[1];  convolution_backward_default_249 = None
        convolution_backward_default_250 = torch.ops.aten.convolution_backward.default(getitem_747, relu_default_18, arg384_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_747 = arg384_1 = None
        getitem_750 = convolution_backward_default_250[0]
        getitem_751 = convolution_backward_default_250[1];  convolution_backward_default_250 = None
        le_scalar_91 = torch.ops.aten.le.Scalar(relu_default_18, 0)
        new_zeros_default_136 = torch.ops.aten.new_zeros.default(getitem_750, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_136 = torch.ops.aten.where.self(le_scalar_91, new_zeros_default_136, getitem_750);  new_zeros_default_136 = getitem_750 = None
        add_tensor_149 = torch.ops.aten.add.Tensor(add_tensor_147, where_self_136);  add_tensor_147 = where_self_136 = None
        sub_tensor_444 = torch.ops.aten.sub.Tensor(arg1259_1, arg1866_1);  arg1259_1 = arg1866_1 = None
        mul_tensor_1275 = torch.ops.aten.mul.Tensor(slice_tensor_68, sub_tensor_444)
        sum_dim_int_list_251 = torch.ops.aten.sum.dim_IntList(mul_tensor_1275, [0, 2, 3]);  mul_tensor_1275 = None
        mul_tensor_1276 = torch.ops.aten.mul.Tensor(sum_dim_int_list_251, 0.0022675736961451248)
        mul_tensor_1277 = torch.ops.aten.mul.Tensor(arg1260_1, arg1260_1)
        mul_tensor_1278 = torch.ops.aten.mul.Tensor(mul_tensor_1276, mul_tensor_1277);  mul_tensor_1276 = mul_tensor_1277 = None
        view_default_408 = torch.ops.aten.view.default(mul_tensor_1278, [1, 336, 1, 1]);  mul_tensor_1278 = None
        mul_tensor_1279 = torch.ops.aten.mul.Tensor(arg1260_1, arg383_1);  arg383_1 = None
        view_default_409 = torch.ops.aten.view.default(mul_tensor_1279, [1, 336, 1, 1]);  mul_tensor_1279 = None
        mul_tensor_1280 = torch.ops.aten.mul.Tensor(sub_tensor_444, view_default_408);  sub_tensor_444 = view_default_408 = None
        sub_tensor_445 = torch.ops.aten.sub.Tensor(slice_tensor_68, mul_tensor_1280);  slice_tensor_68 = mul_tensor_1280 = None
        sub_tensor_446 = torch.ops.aten.sub.Tensor(sub_tensor_445, view_default_402);  sub_tensor_445 = view_default_402 = None
        mul_tensor_1281 = torch.ops.aten.mul.Tensor(sub_tensor_446, view_default_409);  sub_tensor_446 = view_default_409 = None
        mul_tensor_1282 = torch.ops.aten.mul.Tensor(sum_dim_int_list_251, arg1260_1);  sum_dim_int_list_251 = arg1260_1 = None
        convolution_backward_default_251 = torch.ops.aten.convolution_backward.default(mul_tensor_1281, arg1258_1, arg382_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1281 = arg1258_1 = arg382_1 = None
        getitem_753 = convolution_backward_default_251[0]
        getitem_754 = convolution_backward_default_251[1];  convolution_backward_default_251 = None
        convolution_backward_default_252 = torch.ops.aten.convolution_backward.default(getitem_753, arg1257_1, arg381_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_753 = arg381_1 = None
        getitem_756 = convolution_backward_default_252[0]
        getitem_757 = convolution_backward_default_252[1];  convolution_backward_default_252 = None
        le_scalar_92 = torch.ops.aten.le.Scalar(arg1257_1, 0);  arg1257_1 = None
        new_zeros_default_137 = torch.ops.aten.new_zeros.default(getitem_756, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_137 = torch.ops.aten.where.self(le_scalar_92, new_zeros_default_137, getitem_756);  le_scalar_92 = new_zeros_default_137 = getitem_756 = None
        sum_dim_int_list_252 = torch.ops.aten.sum.dim_IntList(where_self_137, [0, 2, 3])
        sub_tensor_447 = torch.ops.aten.sub.Tensor(arg1255_1, arg1867_1);  arg1255_1 = arg1867_1 = None
        mul_tensor_1283 = torch.ops.aten.mul.Tensor(where_self_137, sub_tensor_447)
        sum_dim_int_list_253 = torch.ops.aten.sum.dim_IntList(mul_tensor_1283, [0, 2, 3]);  mul_tensor_1283 = None
        mul_tensor_1284 = torch.ops.aten.mul.Tensor(sum_dim_int_list_252, 0.0022675736961451248);  sum_dim_int_list_252 = None
        view_default_410 = torch.ops.aten.view.default(mul_tensor_1284, [1, 336, 1, 1]);  mul_tensor_1284 = None
        mul_tensor_1285 = torch.ops.aten.mul.Tensor(sum_dim_int_list_253, 0.0022675736961451248)
        mul_tensor_1286 = torch.ops.aten.mul.Tensor(arg1256_1, arg1256_1)
        mul_tensor_1287 = torch.ops.aten.mul.Tensor(mul_tensor_1285, mul_tensor_1286);  mul_tensor_1285 = mul_tensor_1286 = None
        view_default_411 = torch.ops.aten.view.default(mul_tensor_1287, [1, 336, 1, 1]);  mul_tensor_1287 = None
        mul_tensor_1288 = torch.ops.aten.mul.Tensor(arg1256_1, arg380_1);  arg380_1 = None
        view_default_412 = torch.ops.aten.view.default(mul_tensor_1288, [1, 336, 1, 1]);  mul_tensor_1288 = None
        mul_tensor_1289 = torch.ops.aten.mul.Tensor(sub_tensor_447, view_default_411);  sub_tensor_447 = view_default_411 = None
        sub_tensor_448 = torch.ops.aten.sub.Tensor(where_self_137, mul_tensor_1289);  where_self_137 = mul_tensor_1289 = None
        sub_tensor_449 = torch.ops.aten.sub.Tensor(sub_tensor_448, view_default_410);  sub_tensor_448 = view_default_410 = None
        mul_tensor_1290 = torch.ops.aten.mul.Tensor(sub_tensor_449, view_default_412);  sub_tensor_449 = view_default_412 = None
        mul_tensor_1291 = torch.ops.aten.mul.Tensor(sum_dim_int_list_253, arg1256_1);  sum_dim_int_list_253 = arg1256_1 = None
        convolution_backward_default_253 = torch.ops.aten.convolution_backward.default(mul_tensor_1290, arg1254_1, arg379_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1290 = arg1254_1 = arg379_1 = None
        getitem_759 = convolution_backward_default_253[0]
        getitem_760 = convolution_backward_default_253[1];  convolution_backward_default_253 = None
        convolution_backward_default_254 = torch.ops.aten.convolution_backward.default(getitem_759, relu_default_18, arg378_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_759 = arg378_1 = None
        getitem_762 = convolution_backward_default_254[0]
        getitem_763 = convolution_backward_default_254[1];  convolution_backward_default_254 = None
        new_zeros_default_138 = torch.ops.aten.new_zeros.default(getitem_762, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_138 = torch.ops.aten.where.self(le_scalar_91, new_zeros_default_138, getitem_762);  new_zeros_default_138 = getitem_762 = None
        add_tensor_150 = torch.ops.aten.add.Tensor(add_tensor_149, where_self_138);  add_tensor_149 = where_self_138 = None
        sum_dim_int_list_254 = torch.ops.aten.sum.dim_IntList(slice_tensor_67, [0, 2, 3])
        sub_tensor_450 = torch.ops.aten.sub.Tensor(arg1252_1, arg1868_1);  arg1252_1 = arg1868_1 = None
        mul_tensor_1292 = torch.ops.aten.mul.Tensor(slice_tensor_67, sub_tensor_450)
        sum_dim_int_list_255 = torch.ops.aten.sum.dim_IntList(mul_tensor_1292, [0, 2, 3]);  mul_tensor_1292 = None
        mul_tensor_1293 = torch.ops.aten.mul.Tensor(sum_dim_int_list_254, 0.0022675736961451248);  sum_dim_int_list_254 = None
        view_default_413 = torch.ops.aten.view.default(mul_tensor_1293, [1, 336, 1, 1]);  mul_tensor_1293 = None
        mul_tensor_1294 = torch.ops.aten.mul.Tensor(sum_dim_int_list_255, 0.0022675736961451248)
        mul_tensor_1295 = torch.ops.aten.mul.Tensor(arg1253_1, arg1253_1)
        mul_tensor_1296 = torch.ops.aten.mul.Tensor(mul_tensor_1294, mul_tensor_1295);  mul_tensor_1294 = mul_tensor_1295 = None
        view_default_414 = torch.ops.aten.view.default(mul_tensor_1296, [1, 336, 1, 1]);  mul_tensor_1296 = None
        mul_tensor_1297 = torch.ops.aten.mul.Tensor(arg1253_1, arg377_1);  arg377_1 = None
        view_default_415 = torch.ops.aten.view.default(mul_tensor_1297, [1, 336, 1, 1]);  mul_tensor_1297 = None
        mul_tensor_1298 = torch.ops.aten.mul.Tensor(sub_tensor_450, view_default_414);  sub_tensor_450 = view_default_414 = None
        sub_tensor_451 = torch.ops.aten.sub.Tensor(slice_tensor_67, mul_tensor_1298);  mul_tensor_1298 = None
        sub_tensor_452 = torch.ops.aten.sub.Tensor(sub_tensor_451, view_default_413);  sub_tensor_451 = None
        mul_tensor_1299 = torch.ops.aten.mul.Tensor(sub_tensor_452, view_default_415);  sub_tensor_452 = view_default_415 = None
        mul_tensor_1300 = torch.ops.aten.mul.Tensor(sum_dim_int_list_255, arg1253_1);  sum_dim_int_list_255 = arg1253_1 = None
        convolution_backward_default_255 = torch.ops.aten.convolution_backward.default(mul_tensor_1299, arg1251_1, arg376_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1299 = arg1251_1 = arg376_1 = None
        getitem_765 = convolution_backward_default_255[0]
        getitem_766 = convolution_backward_default_255[1];  convolution_backward_default_255 = None
        convolution_backward_default_256 = torch.ops.aten.convolution_backward.default(getitem_765, arg1250_1, arg375_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_765 = arg375_1 = None
        getitem_768 = convolution_backward_default_256[0]
        getitem_769 = convolution_backward_default_256[1];  convolution_backward_default_256 = None
        le_scalar_93 = torch.ops.aten.le.Scalar(arg1250_1, 0);  arg1250_1 = None
        new_zeros_default_139 = torch.ops.aten.new_zeros.default(getitem_768, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_139 = torch.ops.aten.where.self(le_scalar_93, new_zeros_default_139, getitem_768);  le_scalar_93 = new_zeros_default_139 = getitem_768 = None
        sum_dim_int_list_256 = torch.ops.aten.sum.dim_IntList(where_self_139, [0, 2, 3])
        sub_tensor_453 = torch.ops.aten.sub.Tensor(arg1248_1, arg1869_1);  arg1248_1 = arg1869_1 = None
        mul_tensor_1301 = torch.ops.aten.mul.Tensor(where_self_139, sub_tensor_453)
        sum_dim_int_list_257 = torch.ops.aten.sum.dim_IntList(mul_tensor_1301, [0, 2, 3]);  mul_tensor_1301 = None
        mul_tensor_1302 = torch.ops.aten.mul.Tensor(sum_dim_int_list_256, 0.0022675736961451248);  sum_dim_int_list_256 = None
        view_default_416 = torch.ops.aten.view.default(mul_tensor_1302, [1, 336, 1, 1]);  mul_tensor_1302 = None
        mul_tensor_1303 = torch.ops.aten.mul.Tensor(sum_dim_int_list_257, 0.0022675736961451248)
        mul_tensor_1304 = torch.ops.aten.mul.Tensor(arg1249_1, arg1249_1)
        mul_tensor_1305 = torch.ops.aten.mul.Tensor(mul_tensor_1303, mul_tensor_1304);  mul_tensor_1303 = mul_tensor_1304 = None
        view_default_417 = torch.ops.aten.view.default(mul_tensor_1305, [1, 336, 1, 1]);  mul_tensor_1305 = None
        mul_tensor_1306 = torch.ops.aten.mul.Tensor(arg1249_1, arg374_1);  arg374_1 = None
        view_default_418 = torch.ops.aten.view.default(mul_tensor_1306, [1, 336, 1, 1]);  mul_tensor_1306 = None
        mul_tensor_1307 = torch.ops.aten.mul.Tensor(sub_tensor_453, view_default_417);  sub_tensor_453 = view_default_417 = None
        sub_tensor_454 = torch.ops.aten.sub.Tensor(where_self_139, mul_tensor_1307);  where_self_139 = mul_tensor_1307 = None
        sub_tensor_455 = torch.ops.aten.sub.Tensor(sub_tensor_454, view_default_416);  sub_tensor_454 = view_default_416 = None
        mul_tensor_1308 = torch.ops.aten.mul.Tensor(sub_tensor_455, view_default_418);  sub_tensor_455 = view_default_418 = None
        mul_tensor_1309 = torch.ops.aten.mul.Tensor(sum_dim_int_list_257, arg1249_1);  sum_dim_int_list_257 = arg1249_1 = None
        convolution_backward_default_257 = torch.ops.aten.convolution_backward.default(mul_tensor_1308, arg1247_1, arg373_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1308 = arg1247_1 = arg373_1 = None
        getitem_771 = convolution_backward_default_257[0]
        getitem_772 = convolution_backward_default_257[1];  convolution_backward_default_257 = None
        convolution_backward_default_258 = torch.ops.aten.convolution_backward.default(getitem_771, relu_default_18, arg372_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_771 = relu_default_18 = arg372_1 = None
        getitem_774 = convolution_backward_default_258[0]
        getitem_775 = convolution_backward_default_258[1];  convolution_backward_default_258 = None
        new_zeros_default_140 = torch.ops.aten.new_zeros.default(getitem_774, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_140 = torch.ops.aten.where.self(le_scalar_91, new_zeros_default_140, getitem_774);  le_scalar_91 = new_zeros_default_140 = getitem_774 = None
        add_tensor_151 = torch.ops.aten.add.Tensor(add_tensor_150, where_self_140);  add_tensor_150 = where_self_140 = None
        sub_tensor_456 = torch.ops.aten.sub.Tensor(arg1245_1, arg1870_1);  arg1245_1 = arg1870_1 = None
        mul_tensor_1310 = torch.ops.aten.mul.Tensor(slice_tensor_67, sub_tensor_456)
        sum_dim_int_list_258 = torch.ops.aten.sum.dim_IntList(mul_tensor_1310, [0, 2, 3]);  mul_tensor_1310 = None
        mul_tensor_1311 = torch.ops.aten.mul.Tensor(sum_dim_int_list_258, 0.0022675736961451248)
        mul_tensor_1312 = torch.ops.aten.mul.Tensor(arg1246_1, arg1246_1)
        mul_tensor_1313 = torch.ops.aten.mul.Tensor(mul_tensor_1311, mul_tensor_1312);  mul_tensor_1311 = mul_tensor_1312 = None
        view_default_419 = torch.ops.aten.view.default(mul_tensor_1313, [1, 336, 1, 1]);  mul_tensor_1313 = None
        mul_tensor_1314 = torch.ops.aten.mul.Tensor(arg1246_1, arg371_1);  arg371_1 = None
        view_default_420 = torch.ops.aten.view.default(mul_tensor_1314, [1, 336, 1, 1]);  mul_tensor_1314 = None
        mul_tensor_1315 = torch.ops.aten.mul.Tensor(sub_tensor_456, view_default_419);  sub_tensor_456 = view_default_419 = None
        sub_tensor_457 = torch.ops.aten.sub.Tensor(slice_tensor_67, mul_tensor_1315);  slice_tensor_67 = mul_tensor_1315 = None
        sub_tensor_458 = torch.ops.aten.sub.Tensor(sub_tensor_457, view_default_413);  sub_tensor_457 = view_default_413 = None
        mul_tensor_1316 = torch.ops.aten.mul.Tensor(sub_tensor_458, view_default_420);  sub_tensor_458 = view_default_420 = None
        mul_tensor_1317 = torch.ops.aten.mul.Tensor(sum_dim_int_list_258, arg1246_1);  sum_dim_int_list_258 = arg1246_1 = None
        convolution_backward_default_259 = torch.ops.aten.convolution_backward.default(mul_tensor_1316, arg1244_1, arg370_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1316 = arg1244_1 = arg370_1 = None
        getitem_777 = convolution_backward_default_259[0]
        getitem_778 = convolution_backward_default_259[1];  convolution_backward_default_259 = None
        convolution_backward_default_260 = torch.ops.aten.convolution_backward.default(getitem_777, arg1243_1, arg369_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_777 = arg369_1 = None
        getitem_780 = convolution_backward_default_260[0]
        getitem_781 = convolution_backward_default_260[1];  convolution_backward_default_260 = None
        le_scalar_94 = torch.ops.aten.le.Scalar(arg1243_1, 0);  arg1243_1 = None
        new_zeros_default_141 = torch.ops.aten.new_zeros.default(getitem_780, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_141 = torch.ops.aten.where.self(le_scalar_94, new_zeros_default_141, getitem_780);  le_scalar_94 = new_zeros_default_141 = getitem_780 = None
        sum_dim_int_list_259 = torch.ops.aten.sum.dim_IntList(where_self_141, [0, 2, 3])
        sub_tensor_459 = torch.ops.aten.sub.Tensor(arg1241_1, arg1871_1);  arg1241_1 = arg1871_1 = None
        mul_tensor_1318 = torch.ops.aten.mul.Tensor(where_self_141, sub_tensor_459)
        sum_dim_int_list_260 = torch.ops.aten.sum.dim_IntList(mul_tensor_1318, [0, 2, 3]);  mul_tensor_1318 = None
        mul_tensor_1319 = torch.ops.aten.mul.Tensor(sum_dim_int_list_259, 0.0022675736961451248);  sum_dim_int_list_259 = None
        view_default_421 = torch.ops.aten.view.default(mul_tensor_1319, [1, 336, 1, 1]);  mul_tensor_1319 = None
        mul_tensor_1320 = torch.ops.aten.mul.Tensor(sum_dim_int_list_260, 0.0022675736961451248)
        mul_tensor_1321 = torch.ops.aten.mul.Tensor(arg1242_1, arg1242_1)
        mul_tensor_1322 = torch.ops.aten.mul.Tensor(mul_tensor_1320, mul_tensor_1321);  mul_tensor_1320 = mul_tensor_1321 = None
        view_default_422 = torch.ops.aten.view.default(mul_tensor_1322, [1, 336, 1, 1]);  mul_tensor_1322 = None
        mul_tensor_1323 = torch.ops.aten.mul.Tensor(arg1242_1, arg368_1);  arg368_1 = None
        view_default_423 = torch.ops.aten.view.default(mul_tensor_1323, [1, 336, 1, 1]);  mul_tensor_1323 = None
        mul_tensor_1324 = torch.ops.aten.mul.Tensor(sub_tensor_459, view_default_422);  sub_tensor_459 = view_default_422 = None
        sub_tensor_460 = torch.ops.aten.sub.Tensor(where_self_141, mul_tensor_1324);  where_self_141 = mul_tensor_1324 = None
        sub_tensor_461 = torch.ops.aten.sub.Tensor(sub_tensor_460, view_default_421);  sub_tensor_460 = view_default_421 = None
        mul_tensor_1325 = torch.ops.aten.mul.Tensor(sub_tensor_461, view_default_423);  sub_tensor_461 = view_default_423 = None
        mul_tensor_1326 = torch.ops.aten.mul.Tensor(sum_dim_int_list_260, arg1242_1);  sum_dim_int_list_260 = arg1242_1 = None
        convolution_backward_default_261 = torch.ops.aten.convolution_backward.default(mul_tensor_1325, arg1240_1, arg367_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1325 = arg1240_1 = arg367_1 = None
        getitem_783 = convolution_backward_default_261[0]
        getitem_784 = convolution_backward_default_261[1];  convolution_backward_default_261 = None
        convolution_backward_default_262 = torch.ops.aten.convolution_backward.default(getitem_783, relu_default_17, arg366_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_783 = relu_default_17 = arg366_1 = None
        getitem_786 = convolution_backward_default_262[0]
        getitem_787 = convolution_backward_default_262[1];  convolution_backward_default_262 = None
        new_zeros_default_142 = torch.ops.aten.new_zeros.default(getitem_786, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_142 = torch.ops.aten.where.self(le_scalar_89, new_zeros_default_142, getitem_786);  le_scalar_89 = new_zeros_default_142 = getitem_786 = None
        add_tensor_152 = torch.ops.aten.add.Tensor(add_tensor_148, where_self_142);  add_tensor_148 = where_self_142 = None
        view_default_424 = torch.ops.aten.view.default(squeeze_dim_92, [1, 336, 1, 1]);  squeeze_dim_92 = None
        sum_dim_int_list_261 = torch.ops.aten.sum.dim_IntList(add_tensor_152, [0, 2, 3])
        sub_tensor_462 = torch.ops.aten.sub.Tensor(arg1237_1, view_default_424);  arg1237_1 = view_default_424 = None
        mul_tensor_1327 = torch.ops.aten.mul.Tensor(add_tensor_152, sub_tensor_462)
        sum_dim_int_list_262 = torch.ops.aten.sum.dim_IntList(mul_tensor_1327, [0, 2, 3]);  mul_tensor_1327 = None
        mul_tensor_1328 = torch.ops.aten.mul.Tensor(sum_dim_int_list_261, 0.0022675736961451248);  sum_dim_int_list_261 = None
        view_default_425 = torch.ops.aten.view.default(mul_tensor_1328, [1, 336, 1, 1]);  mul_tensor_1328 = None
        mul_tensor_1329 = torch.ops.aten.mul.Tensor(sum_dim_int_list_262, 0.0022675736961451248)
        mul_tensor_1330 = torch.ops.aten.mul.Tensor(squeeze_dim_95, squeeze_dim_95)
        mul_tensor_1331 = torch.ops.aten.mul.Tensor(mul_tensor_1329, mul_tensor_1330);  mul_tensor_1329 = mul_tensor_1330 = None
        view_default_426 = torch.ops.aten.view.default(mul_tensor_1331, [1, 336, 1, 1]);  mul_tensor_1331 = None
        mul_tensor_1332 = torch.ops.aten.mul.Tensor(squeeze_dim_95, arg364_1);  arg364_1 = None
        view_default_427 = torch.ops.aten.view.default(mul_tensor_1332, [1, 336, 1, 1]);  mul_tensor_1332 = None
        mul_tensor_1333 = torch.ops.aten.mul.Tensor(sub_tensor_462, view_default_426);  sub_tensor_462 = view_default_426 = None
        sub_tensor_463 = torch.ops.aten.sub.Tensor(add_tensor_152, mul_tensor_1333);  add_tensor_152 = mul_tensor_1333 = None
        sub_tensor_464 = torch.ops.aten.sub.Tensor(sub_tensor_463, view_default_425);  sub_tensor_463 = view_default_425 = None
        mul_tensor_1334 = torch.ops.aten.mul.Tensor(sub_tensor_464, view_default_427);  sub_tensor_464 = view_default_427 = None
        mul_tensor_1335 = torch.ops.aten.mul.Tensor(sum_dim_int_list_262, squeeze_dim_95);  sum_dim_int_list_262 = squeeze_dim_95 = None
        convolution_backward_default_263 = torch.ops.aten.convolution_backward.default(mul_tensor_1334, arg1236_1, arg363_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1334 = arg1236_1 = arg363_1 = None
        getitem_789 = convolution_backward_default_263[0]
        getitem_790 = convolution_backward_default_263[1];  convolution_backward_default_263 = None
        new_zeros_default_143 = torch.ops.aten.new_zeros.default(getitem_789, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_143 = torch.ops.aten.where.self(le_scalar_87, new_zeros_default_143, getitem_789);  le_scalar_87 = new_zeros_default_143 = getitem_789 = None
        add_tensor_153 = torch.ops.aten.add.Tensor(where_self_132, where_self_143);  where_self_132 = where_self_143 = None
        view_default_428 = torch.ops.aten.view.default(squeeze_dim_86, [1, 336, 1, 1]);  squeeze_dim_86 = None
        sum_dim_int_list_263 = torch.ops.aten.sum.dim_IntList(add_tensor_151, [0, 2, 3])
        sub_tensor_465 = torch.ops.aten.sub.Tensor(arg1233_1, view_default_428);  arg1233_1 = view_default_428 = None
        mul_tensor_1336 = torch.ops.aten.mul.Tensor(add_tensor_151, sub_tensor_465)
        sum_dim_int_list_264 = torch.ops.aten.sum.dim_IntList(mul_tensor_1336, [0, 2, 3]);  mul_tensor_1336 = None
        mul_tensor_1337 = torch.ops.aten.mul.Tensor(sum_dim_int_list_263, 0.0022675736961451248);  sum_dim_int_list_263 = None
        view_default_429 = torch.ops.aten.view.default(mul_tensor_1337, [1, 336, 1, 1]);  mul_tensor_1337 = None
        mul_tensor_1338 = torch.ops.aten.mul.Tensor(sum_dim_int_list_264, 0.0022675736961451248)
        mul_tensor_1339 = torch.ops.aten.mul.Tensor(squeeze_dim_89, squeeze_dim_89)
        mul_tensor_1340 = torch.ops.aten.mul.Tensor(mul_tensor_1338, mul_tensor_1339);  mul_tensor_1338 = mul_tensor_1339 = None
        view_default_430 = torch.ops.aten.view.default(mul_tensor_1340, [1, 336, 1, 1]);  mul_tensor_1340 = None
        mul_tensor_1341 = torch.ops.aten.mul.Tensor(squeeze_dim_89, arg361_1);  arg361_1 = None
        view_default_431 = torch.ops.aten.view.default(mul_tensor_1341, [1, 336, 1, 1]);  mul_tensor_1341 = None
        mul_tensor_1342 = torch.ops.aten.mul.Tensor(sub_tensor_465, view_default_430);  sub_tensor_465 = view_default_430 = None
        sub_tensor_466 = torch.ops.aten.sub.Tensor(add_tensor_151, mul_tensor_1342);  add_tensor_151 = mul_tensor_1342 = None
        sub_tensor_467 = torch.ops.aten.sub.Tensor(sub_tensor_466, view_default_429);  sub_tensor_466 = view_default_429 = None
        mul_tensor_1343 = torch.ops.aten.mul.Tensor(sub_tensor_467, view_default_431);  sub_tensor_467 = view_default_431 = None
        mul_tensor_1344 = torch.ops.aten.mul.Tensor(sum_dim_int_list_264, squeeze_dim_89);  sum_dim_int_list_264 = squeeze_dim_89 = None
        convolution_backward_default_264 = torch.ops.aten.convolution_backward.default(mul_tensor_1343, arg1194_1, arg360_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1343 = arg360_1 = None
        getitem_792 = convolution_backward_default_264[0]
        getitem_793 = convolution_backward_default_264[1];  convolution_backward_default_264 = None
        le_scalar_95 = torch.ops.aten.le.Scalar(arg1194_1, 0)
        new_zeros_default_144 = torch.ops.aten.new_zeros.default(getitem_792, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_144 = torch.ops.aten.where.self(le_scalar_95, new_zeros_default_144, getitem_792);  new_zeros_default_144 = getitem_792 = None
        slice_tensor_72 = torch.ops.aten.slice.Tensor(add_tensor_153, 1, 0, 336)
        slice_tensor_73 = torch.ops.aten.slice.Tensor(add_tensor_153, 1, 336, 672)
        slice_tensor_74 = torch.ops.aten.slice.Tensor(add_tensor_153, 1, 672, 1008)
        slice_tensor_75 = torch.ops.aten.slice.Tensor(add_tensor_153, 1, 1008, 1344)
        slice_tensor_76 = torch.ops.aten.slice.Tensor(add_tensor_153, 1, 1344, 1680)
        slice_tensor_77 = torch.ops.aten.slice.Tensor(add_tensor_153, 1, 1680, 2016);  add_tensor_153 = None
        sum_dim_int_list_265 = torch.ops.aten.sum.dim_IntList(slice_tensor_77, [0, 2, 3])
        sub_tensor_468 = torch.ops.aten.sub.Tensor(arg1231_1, arg1872_1);  arg1231_1 = arg1872_1 = None
        mul_tensor_1345 = torch.ops.aten.mul.Tensor(slice_tensor_77, sub_tensor_468)
        sum_dim_int_list_266 = torch.ops.aten.sum.dim_IntList(mul_tensor_1345, [0, 2, 3]);  mul_tensor_1345 = None
        mul_tensor_1346 = torch.ops.aten.mul.Tensor(sum_dim_int_list_265, 0.0022675736961451248);  sum_dim_int_list_265 = None
        view_default_432 = torch.ops.aten.view.default(mul_tensor_1346, [1, 336, 1, 1]);  mul_tensor_1346 = None
        mul_tensor_1347 = torch.ops.aten.mul.Tensor(sum_dim_int_list_266, 0.0022675736961451248)
        mul_tensor_1348 = torch.ops.aten.mul.Tensor(arg1232_1, arg1232_1)
        mul_tensor_1349 = torch.ops.aten.mul.Tensor(mul_tensor_1347, mul_tensor_1348);  mul_tensor_1347 = mul_tensor_1348 = None
        view_default_433 = torch.ops.aten.view.default(mul_tensor_1349, [1, 336, 1, 1]);  mul_tensor_1349 = None
        mul_tensor_1350 = torch.ops.aten.mul.Tensor(arg1232_1, arg359_1);  arg359_1 = None
        view_default_434 = torch.ops.aten.view.default(mul_tensor_1350, [1, 336, 1, 1]);  mul_tensor_1350 = None
        mul_tensor_1351 = torch.ops.aten.mul.Tensor(sub_tensor_468, view_default_433);  sub_tensor_468 = view_default_433 = None
        sub_tensor_469 = torch.ops.aten.sub.Tensor(slice_tensor_77, mul_tensor_1351);  mul_tensor_1351 = None
        sub_tensor_470 = torch.ops.aten.sub.Tensor(sub_tensor_469, view_default_432);  sub_tensor_469 = view_default_432 = None
        mul_tensor_1352 = torch.ops.aten.mul.Tensor(sub_tensor_470, view_default_434);  sub_tensor_470 = view_default_434 = None
        mul_tensor_1353 = torch.ops.aten.mul.Tensor(sum_dim_int_list_266, arg1232_1);  sum_dim_int_list_266 = arg1232_1 = None
        convolution_backward_default_265 = torch.ops.aten.convolution_backward.default(mul_tensor_1352, arg1230_1, arg358_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1352 = arg1230_1 = arg358_1 = None
        getitem_795 = convolution_backward_default_265[0]
        getitem_796 = convolution_backward_default_265[1];  convolution_backward_default_265 = None
        convolution_backward_default_266 = torch.ops.aten.convolution_backward.default(getitem_795, arg1229_1, arg357_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_795 = arg357_1 = None
        getitem_798 = convolution_backward_default_266[0]
        getitem_799 = convolution_backward_default_266[1];  convolution_backward_default_266 = None
        le_scalar_96 = torch.ops.aten.le.Scalar(arg1229_1, 0);  arg1229_1 = None
        new_zeros_default_145 = torch.ops.aten.new_zeros.default(getitem_798, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_145 = torch.ops.aten.where.self(le_scalar_96, new_zeros_default_145, getitem_798);  le_scalar_96 = new_zeros_default_145 = getitem_798 = None
        sum_dim_int_list_267 = torch.ops.aten.sum.dim_IntList(where_self_145, [0, 2, 3])
        sub_tensor_471 = torch.ops.aten.sub.Tensor(arg1227_1, arg1873_1);  arg1227_1 = arg1873_1 = None
        mul_tensor_1354 = torch.ops.aten.mul.Tensor(where_self_145, sub_tensor_471)
        sum_dim_int_list_268 = torch.ops.aten.sum.dim_IntList(mul_tensor_1354, [0, 2, 3]);  mul_tensor_1354 = None
        mul_tensor_1355 = torch.ops.aten.mul.Tensor(sum_dim_int_list_267, 0.0022675736961451248);  sum_dim_int_list_267 = None
        view_default_435 = torch.ops.aten.view.default(mul_tensor_1355, [1, 336, 1, 1]);  mul_tensor_1355 = None
        mul_tensor_1356 = torch.ops.aten.mul.Tensor(sum_dim_int_list_268, 0.0022675736961451248)
        mul_tensor_1357 = torch.ops.aten.mul.Tensor(arg1228_1, arg1228_1)
        mul_tensor_1358 = torch.ops.aten.mul.Tensor(mul_tensor_1356, mul_tensor_1357);  mul_tensor_1356 = mul_tensor_1357 = None
        view_default_436 = torch.ops.aten.view.default(mul_tensor_1358, [1, 336, 1, 1]);  mul_tensor_1358 = None
        mul_tensor_1359 = torch.ops.aten.mul.Tensor(arg1228_1, arg356_1);  arg356_1 = None
        view_default_437 = torch.ops.aten.view.default(mul_tensor_1359, [1, 336, 1, 1]);  mul_tensor_1359 = None
        mul_tensor_1360 = torch.ops.aten.mul.Tensor(sub_tensor_471, view_default_436);  sub_tensor_471 = view_default_436 = None
        sub_tensor_472 = torch.ops.aten.sub.Tensor(where_self_145, mul_tensor_1360);  where_self_145 = mul_tensor_1360 = None
        sub_tensor_473 = torch.ops.aten.sub.Tensor(sub_tensor_472, view_default_435);  sub_tensor_472 = view_default_435 = None
        mul_tensor_1361 = torch.ops.aten.mul.Tensor(sub_tensor_473, view_default_437);  sub_tensor_473 = view_default_437 = None
        mul_tensor_1362 = torch.ops.aten.mul.Tensor(sum_dim_int_list_268, arg1228_1);  sum_dim_int_list_268 = arg1228_1 = None
        convolution_backward_default_267 = torch.ops.aten.convolution_backward.default(mul_tensor_1361, arg1226_1, arg355_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1361 = arg1226_1 = arg355_1 = None
        getitem_801 = convolution_backward_default_267[0]
        getitem_802 = convolution_backward_default_267[1];  convolution_backward_default_267 = None
        convolution_backward_default_268 = torch.ops.aten.convolution_backward.default(getitem_801, relu_default_15, arg354_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_801 = arg354_1 = None
        getitem_804 = convolution_backward_default_268[0]
        getitem_805 = convolution_backward_default_268[1];  convolution_backward_default_268 = None
        le_scalar_97 = torch.ops.aten.le.Scalar(relu_default_15, 0)
        new_zeros_default_146 = torch.ops.aten.new_zeros.default(getitem_804, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_146 = torch.ops.aten.where.self(le_scalar_97, new_zeros_default_146, getitem_804);  new_zeros_default_146 = getitem_804 = None
        add_tensor_154 = torch.ops.aten.add.Tensor(slice_tensor_77, where_self_146);  slice_tensor_77 = where_self_146 = None
        avg_pool2d_backward_default_26 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_76, add_tensor_12, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_76 = add_tensor_12 = None
        add_tensor_155 = torch.ops.aten.add.Tensor(slice_tensor_72, avg_pool2d_backward_default_26);  slice_tensor_72 = None
        add_tensor_156 = torch.ops.aten.add.Tensor(add_tensor_155, avg_pool2d_backward_default_26);  add_tensor_155 = avg_pool2d_backward_default_26 = None
        add_tensor_157 = torch.ops.aten.add.Tensor(add_tensor_156, slice_tensor_75);  add_tensor_156 = None
        avg_pool2d_backward_default_27 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_75, add_tensor_13, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_75 = add_tensor_13 = None
        add_tensor_158 = torch.ops.aten.add.Tensor(add_tensor_154, avg_pool2d_backward_default_27);  add_tensor_154 = avg_pool2d_backward_default_27 = None
        sum_dim_int_list_269 = torch.ops.aten.sum.dim_IntList(slice_tensor_74, [0, 2, 3])
        sub_tensor_474 = torch.ops.aten.sub.Tensor(arg1224_1, arg1874_1);  arg1224_1 = arg1874_1 = None
        mul_tensor_1363 = torch.ops.aten.mul.Tensor(slice_tensor_74, sub_tensor_474)
        sum_dim_int_list_270 = torch.ops.aten.sum.dim_IntList(mul_tensor_1363, [0, 2, 3]);  mul_tensor_1363 = None
        mul_tensor_1364 = torch.ops.aten.mul.Tensor(sum_dim_int_list_269, 0.0022675736961451248);  sum_dim_int_list_269 = None
        view_default_438 = torch.ops.aten.view.default(mul_tensor_1364, [1, 336, 1, 1]);  mul_tensor_1364 = None
        mul_tensor_1365 = torch.ops.aten.mul.Tensor(sum_dim_int_list_270, 0.0022675736961451248)
        mul_tensor_1366 = torch.ops.aten.mul.Tensor(arg1225_1, arg1225_1)
        mul_tensor_1367 = torch.ops.aten.mul.Tensor(mul_tensor_1365, mul_tensor_1366);  mul_tensor_1365 = mul_tensor_1366 = None
        view_default_439 = torch.ops.aten.view.default(mul_tensor_1367, [1, 336, 1, 1]);  mul_tensor_1367 = None
        mul_tensor_1368 = torch.ops.aten.mul.Tensor(arg1225_1, arg353_1);  arg353_1 = None
        view_default_440 = torch.ops.aten.view.default(mul_tensor_1368, [1, 336, 1, 1]);  mul_tensor_1368 = None
        mul_tensor_1369 = torch.ops.aten.mul.Tensor(sub_tensor_474, view_default_439);  sub_tensor_474 = view_default_439 = None
        sub_tensor_475 = torch.ops.aten.sub.Tensor(slice_tensor_74, mul_tensor_1369);  mul_tensor_1369 = None
        sub_tensor_476 = torch.ops.aten.sub.Tensor(sub_tensor_475, view_default_438);  sub_tensor_475 = None
        mul_tensor_1370 = torch.ops.aten.mul.Tensor(sub_tensor_476, view_default_440);  sub_tensor_476 = view_default_440 = None
        mul_tensor_1371 = torch.ops.aten.mul.Tensor(sum_dim_int_list_270, arg1225_1);  sum_dim_int_list_270 = arg1225_1 = None
        convolution_backward_default_269 = torch.ops.aten.convolution_backward.default(mul_tensor_1370, arg1223_1, arg352_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1370 = arg1223_1 = arg352_1 = None
        getitem_807 = convolution_backward_default_269[0]
        getitem_808 = convolution_backward_default_269[1];  convolution_backward_default_269 = None
        convolution_backward_default_270 = torch.ops.aten.convolution_backward.default(getitem_807, arg1222_1, arg351_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_807 = arg351_1 = None
        getitem_810 = convolution_backward_default_270[0]
        getitem_811 = convolution_backward_default_270[1];  convolution_backward_default_270 = None
        le_scalar_98 = torch.ops.aten.le.Scalar(arg1222_1, 0);  arg1222_1 = None
        new_zeros_default_147 = torch.ops.aten.new_zeros.default(getitem_810, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_147 = torch.ops.aten.where.self(le_scalar_98, new_zeros_default_147, getitem_810);  le_scalar_98 = new_zeros_default_147 = getitem_810 = None
        sum_dim_int_list_271 = torch.ops.aten.sum.dim_IntList(where_self_147, [0, 2, 3])
        sub_tensor_477 = torch.ops.aten.sub.Tensor(arg1220_1, arg1875_1);  arg1220_1 = arg1875_1 = None
        mul_tensor_1372 = torch.ops.aten.mul.Tensor(where_self_147, sub_tensor_477)
        sum_dim_int_list_272 = torch.ops.aten.sum.dim_IntList(mul_tensor_1372, [0, 2, 3]);  mul_tensor_1372 = None
        mul_tensor_1373 = torch.ops.aten.mul.Tensor(sum_dim_int_list_271, 0.0022675736961451248);  sum_dim_int_list_271 = None
        view_default_441 = torch.ops.aten.view.default(mul_tensor_1373, [1, 336, 1, 1]);  mul_tensor_1373 = None
        mul_tensor_1374 = torch.ops.aten.mul.Tensor(sum_dim_int_list_272, 0.0022675736961451248)
        mul_tensor_1375 = torch.ops.aten.mul.Tensor(arg1221_1, arg1221_1)
        mul_tensor_1376 = torch.ops.aten.mul.Tensor(mul_tensor_1374, mul_tensor_1375);  mul_tensor_1374 = mul_tensor_1375 = None
        view_default_442 = torch.ops.aten.view.default(mul_tensor_1376, [1, 336, 1, 1]);  mul_tensor_1376 = None
        mul_tensor_1377 = torch.ops.aten.mul.Tensor(arg1221_1, arg350_1);  arg350_1 = None
        view_default_443 = torch.ops.aten.view.default(mul_tensor_1377, [1, 336, 1, 1]);  mul_tensor_1377 = None
        mul_tensor_1378 = torch.ops.aten.mul.Tensor(sub_tensor_477, view_default_442);  sub_tensor_477 = view_default_442 = None
        sub_tensor_478 = torch.ops.aten.sub.Tensor(where_self_147, mul_tensor_1378);  where_self_147 = mul_tensor_1378 = None
        sub_tensor_479 = torch.ops.aten.sub.Tensor(sub_tensor_478, view_default_441);  sub_tensor_478 = view_default_441 = None
        mul_tensor_1379 = torch.ops.aten.mul.Tensor(sub_tensor_479, view_default_443);  sub_tensor_479 = view_default_443 = None
        mul_tensor_1380 = torch.ops.aten.mul.Tensor(sum_dim_int_list_272, arg1221_1);  sum_dim_int_list_272 = arg1221_1 = None
        convolution_backward_default_271 = torch.ops.aten.convolution_backward.default(mul_tensor_1379, arg1219_1, arg349_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1379 = arg1219_1 = arg349_1 = None
        getitem_813 = convolution_backward_default_271[0]
        getitem_814 = convolution_backward_default_271[1];  convolution_backward_default_271 = None
        convolution_backward_default_272 = torch.ops.aten.convolution_backward.default(getitem_813, relu_default_16, arg348_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_813 = arg348_1 = None
        getitem_816 = convolution_backward_default_272[0]
        getitem_817 = convolution_backward_default_272[1];  convolution_backward_default_272 = None
        le_scalar_99 = torch.ops.aten.le.Scalar(relu_default_16, 0)
        new_zeros_default_148 = torch.ops.aten.new_zeros.default(getitem_816, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_148 = torch.ops.aten.where.self(le_scalar_99, new_zeros_default_148, getitem_816);  new_zeros_default_148 = getitem_816 = None
        add_tensor_159 = torch.ops.aten.add.Tensor(add_tensor_157, where_self_148);  add_tensor_157 = where_self_148 = None
        sub_tensor_480 = torch.ops.aten.sub.Tensor(arg1217_1, arg1876_1);  arg1217_1 = arg1876_1 = None
        mul_tensor_1381 = torch.ops.aten.mul.Tensor(slice_tensor_74, sub_tensor_480)
        sum_dim_int_list_273 = torch.ops.aten.sum.dim_IntList(mul_tensor_1381, [0, 2, 3]);  mul_tensor_1381 = None
        mul_tensor_1382 = torch.ops.aten.mul.Tensor(sum_dim_int_list_273, 0.0022675736961451248)
        mul_tensor_1383 = torch.ops.aten.mul.Tensor(arg1218_1, arg1218_1)
        mul_tensor_1384 = torch.ops.aten.mul.Tensor(mul_tensor_1382, mul_tensor_1383);  mul_tensor_1382 = mul_tensor_1383 = None
        view_default_444 = torch.ops.aten.view.default(mul_tensor_1384, [1, 336, 1, 1]);  mul_tensor_1384 = None
        mul_tensor_1385 = torch.ops.aten.mul.Tensor(arg1218_1, arg347_1);  arg347_1 = None
        view_default_445 = torch.ops.aten.view.default(mul_tensor_1385, [1, 336, 1, 1]);  mul_tensor_1385 = None
        mul_tensor_1386 = torch.ops.aten.mul.Tensor(sub_tensor_480, view_default_444);  sub_tensor_480 = view_default_444 = None
        sub_tensor_481 = torch.ops.aten.sub.Tensor(slice_tensor_74, mul_tensor_1386);  slice_tensor_74 = mul_tensor_1386 = None
        sub_tensor_482 = torch.ops.aten.sub.Tensor(sub_tensor_481, view_default_438);  sub_tensor_481 = view_default_438 = None
        mul_tensor_1387 = torch.ops.aten.mul.Tensor(sub_tensor_482, view_default_445);  sub_tensor_482 = view_default_445 = None
        mul_tensor_1388 = torch.ops.aten.mul.Tensor(sum_dim_int_list_273, arg1218_1);  sum_dim_int_list_273 = arg1218_1 = None
        convolution_backward_default_273 = torch.ops.aten.convolution_backward.default(mul_tensor_1387, arg1216_1, arg346_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1387 = arg1216_1 = arg346_1 = None
        getitem_819 = convolution_backward_default_273[0]
        getitem_820 = convolution_backward_default_273[1];  convolution_backward_default_273 = None
        convolution_backward_default_274 = torch.ops.aten.convolution_backward.default(getitem_819, arg1215_1, arg345_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_819 = arg345_1 = None
        getitem_822 = convolution_backward_default_274[0]
        getitem_823 = convolution_backward_default_274[1];  convolution_backward_default_274 = None
        le_scalar_100 = torch.ops.aten.le.Scalar(arg1215_1, 0);  arg1215_1 = None
        new_zeros_default_149 = torch.ops.aten.new_zeros.default(getitem_822, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_149 = torch.ops.aten.where.self(le_scalar_100, new_zeros_default_149, getitem_822);  le_scalar_100 = new_zeros_default_149 = getitem_822 = None
        sum_dim_int_list_274 = torch.ops.aten.sum.dim_IntList(where_self_149, [0, 2, 3])
        sub_tensor_483 = torch.ops.aten.sub.Tensor(arg1213_1, arg1877_1);  arg1213_1 = arg1877_1 = None
        mul_tensor_1389 = torch.ops.aten.mul.Tensor(where_self_149, sub_tensor_483)
        sum_dim_int_list_275 = torch.ops.aten.sum.dim_IntList(mul_tensor_1389, [0, 2, 3]);  mul_tensor_1389 = None
        mul_tensor_1390 = torch.ops.aten.mul.Tensor(sum_dim_int_list_274, 0.0022675736961451248);  sum_dim_int_list_274 = None
        view_default_446 = torch.ops.aten.view.default(mul_tensor_1390, [1, 336, 1, 1]);  mul_tensor_1390 = None
        mul_tensor_1391 = torch.ops.aten.mul.Tensor(sum_dim_int_list_275, 0.0022675736961451248)
        mul_tensor_1392 = torch.ops.aten.mul.Tensor(arg1214_1, arg1214_1)
        mul_tensor_1393 = torch.ops.aten.mul.Tensor(mul_tensor_1391, mul_tensor_1392);  mul_tensor_1391 = mul_tensor_1392 = None
        view_default_447 = torch.ops.aten.view.default(mul_tensor_1393, [1, 336, 1, 1]);  mul_tensor_1393 = None
        mul_tensor_1394 = torch.ops.aten.mul.Tensor(arg1214_1, arg344_1);  arg344_1 = None
        view_default_448 = torch.ops.aten.view.default(mul_tensor_1394, [1, 336, 1, 1]);  mul_tensor_1394 = None
        mul_tensor_1395 = torch.ops.aten.mul.Tensor(sub_tensor_483, view_default_447);  sub_tensor_483 = view_default_447 = None
        sub_tensor_484 = torch.ops.aten.sub.Tensor(where_self_149, mul_tensor_1395);  where_self_149 = mul_tensor_1395 = None
        sub_tensor_485 = torch.ops.aten.sub.Tensor(sub_tensor_484, view_default_446);  sub_tensor_484 = view_default_446 = None
        mul_tensor_1396 = torch.ops.aten.mul.Tensor(sub_tensor_485, view_default_448);  sub_tensor_485 = view_default_448 = None
        mul_tensor_1397 = torch.ops.aten.mul.Tensor(sum_dim_int_list_275, arg1214_1);  sum_dim_int_list_275 = arg1214_1 = None
        convolution_backward_default_275 = torch.ops.aten.convolution_backward.default(mul_tensor_1396, arg1212_1, arg343_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1396 = arg1212_1 = arg343_1 = None
        getitem_825 = convolution_backward_default_275[0]
        getitem_826 = convolution_backward_default_275[1];  convolution_backward_default_275 = None
        convolution_backward_default_276 = torch.ops.aten.convolution_backward.default(getitem_825, relu_default_16, arg342_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_825 = arg342_1 = None
        getitem_828 = convolution_backward_default_276[0]
        getitem_829 = convolution_backward_default_276[1];  convolution_backward_default_276 = None
        new_zeros_default_150 = torch.ops.aten.new_zeros.default(getitem_828, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_150 = torch.ops.aten.where.self(le_scalar_99, new_zeros_default_150, getitem_828);  new_zeros_default_150 = getitem_828 = None
        add_tensor_160 = torch.ops.aten.add.Tensor(add_tensor_159, where_self_150);  add_tensor_159 = where_self_150 = None
        sum_dim_int_list_276 = torch.ops.aten.sum.dim_IntList(slice_tensor_73, [0, 2, 3])
        sub_tensor_486 = torch.ops.aten.sub.Tensor(arg1210_1, arg1878_1);  arg1210_1 = arg1878_1 = None
        mul_tensor_1398 = torch.ops.aten.mul.Tensor(slice_tensor_73, sub_tensor_486)
        sum_dim_int_list_277 = torch.ops.aten.sum.dim_IntList(mul_tensor_1398, [0, 2, 3]);  mul_tensor_1398 = None
        mul_tensor_1399 = torch.ops.aten.mul.Tensor(sum_dim_int_list_276, 0.0022675736961451248);  sum_dim_int_list_276 = None
        view_default_449 = torch.ops.aten.view.default(mul_tensor_1399, [1, 336, 1, 1]);  mul_tensor_1399 = None
        mul_tensor_1400 = torch.ops.aten.mul.Tensor(sum_dim_int_list_277, 0.0022675736961451248)
        mul_tensor_1401 = torch.ops.aten.mul.Tensor(arg1211_1, arg1211_1)
        mul_tensor_1402 = torch.ops.aten.mul.Tensor(mul_tensor_1400, mul_tensor_1401);  mul_tensor_1400 = mul_tensor_1401 = None
        view_default_450 = torch.ops.aten.view.default(mul_tensor_1402, [1, 336, 1, 1]);  mul_tensor_1402 = None
        mul_tensor_1403 = torch.ops.aten.mul.Tensor(arg1211_1, arg341_1);  arg341_1 = None
        view_default_451 = torch.ops.aten.view.default(mul_tensor_1403, [1, 336, 1, 1]);  mul_tensor_1403 = None
        mul_tensor_1404 = torch.ops.aten.mul.Tensor(sub_tensor_486, view_default_450);  sub_tensor_486 = view_default_450 = None
        sub_tensor_487 = torch.ops.aten.sub.Tensor(slice_tensor_73, mul_tensor_1404);  mul_tensor_1404 = None
        sub_tensor_488 = torch.ops.aten.sub.Tensor(sub_tensor_487, view_default_449);  sub_tensor_487 = None
        mul_tensor_1405 = torch.ops.aten.mul.Tensor(sub_tensor_488, view_default_451);  sub_tensor_488 = view_default_451 = None
        mul_tensor_1406 = torch.ops.aten.mul.Tensor(sum_dim_int_list_277, arg1211_1);  sum_dim_int_list_277 = arg1211_1 = None
        convolution_backward_default_277 = torch.ops.aten.convolution_backward.default(mul_tensor_1405, arg1209_1, arg340_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1405 = arg1209_1 = arg340_1 = None
        getitem_831 = convolution_backward_default_277[0]
        getitem_832 = convolution_backward_default_277[1];  convolution_backward_default_277 = None
        convolution_backward_default_278 = torch.ops.aten.convolution_backward.default(getitem_831, arg1208_1, arg339_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_831 = arg339_1 = None
        getitem_834 = convolution_backward_default_278[0]
        getitem_835 = convolution_backward_default_278[1];  convolution_backward_default_278 = None
        le_scalar_101 = torch.ops.aten.le.Scalar(arg1208_1, 0);  arg1208_1 = None
        new_zeros_default_151 = torch.ops.aten.new_zeros.default(getitem_834, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_151 = torch.ops.aten.where.self(le_scalar_101, new_zeros_default_151, getitem_834);  le_scalar_101 = new_zeros_default_151 = getitem_834 = None
        sum_dim_int_list_278 = torch.ops.aten.sum.dim_IntList(where_self_151, [0, 2, 3])
        sub_tensor_489 = torch.ops.aten.sub.Tensor(arg1206_1, arg1879_1);  arg1206_1 = arg1879_1 = None
        mul_tensor_1407 = torch.ops.aten.mul.Tensor(where_self_151, sub_tensor_489)
        sum_dim_int_list_279 = torch.ops.aten.sum.dim_IntList(mul_tensor_1407, [0, 2, 3]);  mul_tensor_1407 = None
        mul_tensor_1408 = torch.ops.aten.mul.Tensor(sum_dim_int_list_278, 0.0022675736961451248);  sum_dim_int_list_278 = None
        view_default_452 = torch.ops.aten.view.default(mul_tensor_1408, [1, 336, 1, 1]);  mul_tensor_1408 = None
        mul_tensor_1409 = torch.ops.aten.mul.Tensor(sum_dim_int_list_279, 0.0022675736961451248)
        mul_tensor_1410 = torch.ops.aten.mul.Tensor(arg1207_1, arg1207_1)
        mul_tensor_1411 = torch.ops.aten.mul.Tensor(mul_tensor_1409, mul_tensor_1410);  mul_tensor_1409 = mul_tensor_1410 = None
        view_default_453 = torch.ops.aten.view.default(mul_tensor_1411, [1, 336, 1, 1]);  mul_tensor_1411 = None
        mul_tensor_1412 = torch.ops.aten.mul.Tensor(arg1207_1, arg338_1);  arg338_1 = None
        view_default_454 = torch.ops.aten.view.default(mul_tensor_1412, [1, 336, 1, 1]);  mul_tensor_1412 = None
        mul_tensor_1413 = torch.ops.aten.mul.Tensor(sub_tensor_489, view_default_453);  sub_tensor_489 = view_default_453 = None
        sub_tensor_490 = torch.ops.aten.sub.Tensor(where_self_151, mul_tensor_1413);  where_self_151 = mul_tensor_1413 = None
        sub_tensor_491 = torch.ops.aten.sub.Tensor(sub_tensor_490, view_default_452);  sub_tensor_490 = view_default_452 = None
        mul_tensor_1414 = torch.ops.aten.mul.Tensor(sub_tensor_491, view_default_454);  sub_tensor_491 = view_default_454 = None
        mul_tensor_1415 = torch.ops.aten.mul.Tensor(sum_dim_int_list_279, arg1207_1);  sum_dim_int_list_279 = arg1207_1 = None
        convolution_backward_default_279 = torch.ops.aten.convolution_backward.default(mul_tensor_1414, arg1205_1, arg337_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1414 = arg1205_1 = arg337_1 = None
        getitem_837 = convolution_backward_default_279[0]
        getitem_838 = convolution_backward_default_279[1];  convolution_backward_default_279 = None
        convolution_backward_default_280 = torch.ops.aten.convolution_backward.default(getitem_837, relu_default_16, arg336_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_837 = relu_default_16 = arg336_1 = None
        getitem_840 = convolution_backward_default_280[0]
        getitem_841 = convolution_backward_default_280[1];  convolution_backward_default_280 = None
        new_zeros_default_152 = torch.ops.aten.new_zeros.default(getitem_840, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_152 = torch.ops.aten.where.self(le_scalar_99, new_zeros_default_152, getitem_840);  le_scalar_99 = new_zeros_default_152 = getitem_840 = None
        add_tensor_161 = torch.ops.aten.add.Tensor(add_tensor_160, where_self_152);  add_tensor_160 = where_self_152 = None
        sub_tensor_492 = torch.ops.aten.sub.Tensor(arg1203_1, arg1880_1);  arg1203_1 = arg1880_1 = None
        mul_tensor_1416 = torch.ops.aten.mul.Tensor(slice_tensor_73, sub_tensor_492)
        sum_dim_int_list_280 = torch.ops.aten.sum.dim_IntList(mul_tensor_1416, [0, 2, 3]);  mul_tensor_1416 = None
        mul_tensor_1417 = torch.ops.aten.mul.Tensor(sum_dim_int_list_280, 0.0022675736961451248)
        mul_tensor_1418 = torch.ops.aten.mul.Tensor(arg1204_1, arg1204_1)
        mul_tensor_1419 = torch.ops.aten.mul.Tensor(mul_tensor_1417, mul_tensor_1418);  mul_tensor_1417 = mul_tensor_1418 = None
        view_default_455 = torch.ops.aten.view.default(mul_tensor_1419, [1, 336, 1, 1]);  mul_tensor_1419 = None
        mul_tensor_1420 = torch.ops.aten.mul.Tensor(arg1204_1, arg335_1);  arg335_1 = None
        view_default_456 = torch.ops.aten.view.default(mul_tensor_1420, [1, 336, 1, 1]);  mul_tensor_1420 = None
        mul_tensor_1421 = torch.ops.aten.mul.Tensor(sub_tensor_492, view_default_455);  sub_tensor_492 = view_default_455 = None
        sub_tensor_493 = torch.ops.aten.sub.Tensor(slice_tensor_73, mul_tensor_1421);  slice_tensor_73 = mul_tensor_1421 = None
        sub_tensor_494 = torch.ops.aten.sub.Tensor(sub_tensor_493, view_default_449);  sub_tensor_493 = view_default_449 = None
        mul_tensor_1422 = torch.ops.aten.mul.Tensor(sub_tensor_494, view_default_456);  sub_tensor_494 = view_default_456 = None
        mul_tensor_1423 = torch.ops.aten.mul.Tensor(sum_dim_int_list_280, arg1204_1);  sum_dim_int_list_280 = arg1204_1 = None
        convolution_backward_default_281 = torch.ops.aten.convolution_backward.default(mul_tensor_1422, arg1202_1, arg334_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1422 = arg1202_1 = arg334_1 = None
        getitem_843 = convolution_backward_default_281[0]
        getitem_844 = convolution_backward_default_281[1];  convolution_backward_default_281 = None
        convolution_backward_default_282 = torch.ops.aten.convolution_backward.default(getitem_843, arg1201_1, arg333_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_843 = arg333_1 = None
        getitem_846 = convolution_backward_default_282[0]
        getitem_847 = convolution_backward_default_282[1];  convolution_backward_default_282 = None
        le_scalar_102 = torch.ops.aten.le.Scalar(arg1201_1, 0);  arg1201_1 = None
        new_zeros_default_153 = torch.ops.aten.new_zeros.default(getitem_846, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_153 = torch.ops.aten.where.self(le_scalar_102, new_zeros_default_153, getitem_846);  le_scalar_102 = new_zeros_default_153 = getitem_846 = None
        sum_dim_int_list_281 = torch.ops.aten.sum.dim_IntList(where_self_153, [0, 2, 3])
        sub_tensor_495 = torch.ops.aten.sub.Tensor(arg1199_1, arg1881_1);  arg1199_1 = arg1881_1 = None
        mul_tensor_1424 = torch.ops.aten.mul.Tensor(where_self_153, sub_tensor_495)
        sum_dim_int_list_282 = torch.ops.aten.sum.dim_IntList(mul_tensor_1424, [0, 2, 3]);  mul_tensor_1424 = None
        mul_tensor_1425 = torch.ops.aten.mul.Tensor(sum_dim_int_list_281, 0.0022675736961451248);  sum_dim_int_list_281 = None
        view_default_457 = torch.ops.aten.view.default(mul_tensor_1425, [1, 336, 1, 1]);  mul_tensor_1425 = None
        mul_tensor_1426 = torch.ops.aten.mul.Tensor(sum_dim_int_list_282, 0.0022675736961451248)
        mul_tensor_1427 = torch.ops.aten.mul.Tensor(arg1200_1, arg1200_1)
        mul_tensor_1428 = torch.ops.aten.mul.Tensor(mul_tensor_1426, mul_tensor_1427);  mul_tensor_1426 = mul_tensor_1427 = None
        view_default_458 = torch.ops.aten.view.default(mul_tensor_1428, [1, 336, 1, 1]);  mul_tensor_1428 = None
        mul_tensor_1429 = torch.ops.aten.mul.Tensor(arg1200_1, arg332_1);  arg332_1 = None
        view_default_459 = torch.ops.aten.view.default(mul_tensor_1429, [1, 336, 1, 1]);  mul_tensor_1429 = None
        mul_tensor_1430 = torch.ops.aten.mul.Tensor(sub_tensor_495, view_default_458);  sub_tensor_495 = view_default_458 = None
        sub_tensor_496 = torch.ops.aten.sub.Tensor(where_self_153, mul_tensor_1430);  where_self_153 = mul_tensor_1430 = None
        sub_tensor_497 = torch.ops.aten.sub.Tensor(sub_tensor_496, view_default_457);  sub_tensor_496 = view_default_457 = None
        mul_tensor_1431 = torch.ops.aten.mul.Tensor(sub_tensor_497, view_default_459);  sub_tensor_497 = view_default_459 = None
        mul_tensor_1432 = torch.ops.aten.mul.Tensor(sum_dim_int_list_282, arg1200_1);  sum_dim_int_list_282 = arg1200_1 = None
        convolution_backward_default_283 = torch.ops.aten.convolution_backward.default(mul_tensor_1431, arg1198_1, arg331_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1431 = arg1198_1 = arg331_1 = None
        getitem_849 = convolution_backward_default_283[0]
        getitem_850 = convolution_backward_default_283[1];  convolution_backward_default_283 = None
        convolution_backward_default_284 = torch.ops.aten.convolution_backward.default(getitem_849, relu_default_15, arg330_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_849 = relu_default_15 = arg330_1 = None
        getitem_852 = convolution_backward_default_284[0]
        getitem_853 = convolution_backward_default_284[1];  convolution_backward_default_284 = None
        new_zeros_default_154 = torch.ops.aten.new_zeros.default(getitem_852, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_154 = torch.ops.aten.where.self(le_scalar_97, new_zeros_default_154, getitem_852);  le_scalar_97 = new_zeros_default_154 = getitem_852 = None
        add_tensor_162 = torch.ops.aten.add.Tensor(add_tensor_158, where_self_154);  add_tensor_158 = where_self_154 = None
        view_default_460 = torch.ops.aten.view.default(squeeze_dim_80, [1, 336, 1, 1]);  squeeze_dim_80 = None
        sum_dim_int_list_283 = torch.ops.aten.sum.dim_IntList(add_tensor_162, [0, 2, 3])
        sub_tensor_498 = torch.ops.aten.sub.Tensor(arg1195_1, view_default_460);  arg1195_1 = view_default_460 = None
        mul_tensor_1433 = torch.ops.aten.mul.Tensor(add_tensor_162, sub_tensor_498)
        sum_dim_int_list_284 = torch.ops.aten.sum.dim_IntList(mul_tensor_1433, [0, 2, 3]);  mul_tensor_1433 = None
        mul_tensor_1434 = torch.ops.aten.mul.Tensor(sum_dim_int_list_283, 0.0022675736961451248);  sum_dim_int_list_283 = None
        view_default_461 = torch.ops.aten.view.default(mul_tensor_1434, [1, 336, 1, 1]);  mul_tensor_1434 = None
        mul_tensor_1435 = torch.ops.aten.mul.Tensor(sum_dim_int_list_284, 0.0022675736961451248)
        mul_tensor_1436 = torch.ops.aten.mul.Tensor(squeeze_dim_83, squeeze_dim_83)
        mul_tensor_1437 = torch.ops.aten.mul.Tensor(mul_tensor_1435, mul_tensor_1436);  mul_tensor_1435 = mul_tensor_1436 = None
        view_default_462 = torch.ops.aten.view.default(mul_tensor_1437, [1, 336, 1, 1]);  mul_tensor_1437 = None
        mul_tensor_1438 = torch.ops.aten.mul.Tensor(squeeze_dim_83, arg328_1);  arg328_1 = None
        view_default_463 = torch.ops.aten.view.default(mul_tensor_1438, [1, 336, 1, 1]);  mul_tensor_1438 = None
        mul_tensor_1439 = torch.ops.aten.mul.Tensor(sub_tensor_498, view_default_462);  sub_tensor_498 = view_default_462 = None
        sub_tensor_499 = torch.ops.aten.sub.Tensor(add_tensor_162, mul_tensor_1439);  add_tensor_162 = mul_tensor_1439 = None
        sub_tensor_500 = torch.ops.aten.sub.Tensor(sub_tensor_499, view_default_461);  sub_tensor_499 = view_default_461 = None
        mul_tensor_1440 = torch.ops.aten.mul.Tensor(sub_tensor_500, view_default_463);  sub_tensor_500 = view_default_463 = None
        mul_tensor_1441 = torch.ops.aten.mul.Tensor(sum_dim_int_list_284, squeeze_dim_83);  sum_dim_int_list_284 = squeeze_dim_83 = None
        convolution_backward_default_285 = torch.ops.aten.convolution_backward.default(mul_tensor_1440, arg1194_1, arg327_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1440 = arg1194_1 = arg327_1 = None
        getitem_855 = convolution_backward_default_285[0]
        getitem_856 = convolution_backward_default_285[1];  convolution_backward_default_285 = None
        new_zeros_default_155 = torch.ops.aten.new_zeros.default(getitem_855, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_155 = torch.ops.aten.where.self(le_scalar_95, new_zeros_default_155, getitem_855);  le_scalar_95 = new_zeros_default_155 = getitem_855 = None
        add_tensor_163 = torch.ops.aten.add.Tensor(where_self_144, where_self_155);  where_self_144 = where_self_155 = None
        view_default_464 = torch.ops.aten.view.default(squeeze_dim_74, [1, 336, 1, 1]);  squeeze_dim_74 = None
        sum_dim_int_list_285 = torch.ops.aten.sum.dim_IntList(add_tensor_161, [0, 2, 3])
        sub_tensor_501 = torch.ops.aten.sub.Tensor(arg1191_1, view_default_464);  arg1191_1 = view_default_464 = None
        mul_tensor_1442 = torch.ops.aten.mul.Tensor(add_tensor_161, sub_tensor_501)
        sum_dim_int_list_286 = torch.ops.aten.sum.dim_IntList(mul_tensor_1442, [0, 2, 3]);  mul_tensor_1442 = None
        mul_tensor_1443 = torch.ops.aten.mul.Tensor(sum_dim_int_list_285, 0.0022675736961451248);  sum_dim_int_list_285 = None
        view_default_465 = torch.ops.aten.view.default(mul_tensor_1443, [1, 336, 1, 1]);  mul_tensor_1443 = None
        mul_tensor_1444 = torch.ops.aten.mul.Tensor(sum_dim_int_list_286, 0.0022675736961451248)
        mul_tensor_1445 = torch.ops.aten.mul.Tensor(squeeze_dim_77, squeeze_dim_77)
        mul_tensor_1446 = torch.ops.aten.mul.Tensor(mul_tensor_1444, mul_tensor_1445);  mul_tensor_1444 = mul_tensor_1445 = None
        view_default_466 = torch.ops.aten.view.default(mul_tensor_1446, [1, 336, 1, 1]);  mul_tensor_1446 = None
        mul_tensor_1447 = torch.ops.aten.mul.Tensor(squeeze_dim_77, arg325_1);  arg325_1 = None
        view_default_467 = torch.ops.aten.view.default(mul_tensor_1447, [1, 336, 1, 1]);  mul_tensor_1447 = None
        mul_tensor_1448 = torch.ops.aten.mul.Tensor(sub_tensor_501, view_default_466);  sub_tensor_501 = view_default_466 = None
        sub_tensor_502 = torch.ops.aten.sub.Tensor(add_tensor_161, mul_tensor_1448);  add_tensor_161 = mul_tensor_1448 = None
        sub_tensor_503 = torch.ops.aten.sub.Tensor(sub_tensor_502, view_default_465);  sub_tensor_502 = view_default_465 = None
        mul_tensor_1449 = torch.ops.aten.mul.Tensor(sub_tensor_503, view_default_467);  sub_tensor_503 = view_default_467 = None
        mul_tensor_1450 = torch.ops.aten.mul.Tensor(sum_dim_int_list_286, squeeze_dim_77);  sum_dim_int_list_286 = squeeze_dim_77 = None
        slice_tensor_78 = torch.ops.aten.slice.Tensor(mul_tensor_1449, 1, 0, 168)
        slice_tensor_79 = torch.ops.aten.slice.Tensor(mul_tensor_1449, 1, 168, 336);  mul_tensor_1449 = None
        convolution_backward_default_286 = torch.ops.aten.convolution_backward.default(slice_tensor_79, arg1190_1, arg324_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_tensor_79 = arg1190_1 = arg324_1 = None
        getitem_858 = convolution_backward_default_286[0]
        getitem_859 = convolution_backward_default_286[1];  convolution_backward_default_286 = None
        avg_pool2d_backward_default_28 = torch.ops.aten.avg_pool2d_backward.default(getitem_858, arg1189_1, [1, 1], [2, 2], [0, 0], False, False, None);  getitem_858 = arg1189_1 = None
        constant_pad_nd_default_8 = torch.ops.aten.constant_pad_nd.default(avg_pool2d_backward_default_28, [1, -1, 1, -1]);  avg_pool2d_backward_default_28 = None
        convolution_backward_default_287 = torch.ops.aten.convolution_backward.default(slice_tensor_78, arg1188_1, arg323_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_tensor_78 = arg1188_1 = arg323_1 = None
        getitem_861 = convolution_backward_default_287[0]
        getitem_862 = convolution_backward_default_287[1];  convolution_backward_default_287 = None
        avg_pool2d_backward_default_29 = torch.ops.aten.avg_pool2d_backward.default(getitem_861, arg1102_1, [1, 1], [2, 2], [0, 0], False, False, None);  getitem_861 = None
        add_tensor_164 = torch.ops.aten.add.Tensor(constant_pad_nd_default_8, avg_pool2d_backward_default_29);  constant_pad_nd_default_8 = avg_pool2d_backward_default_29 = None
        le_scalar_103 = torch.ops.aten.le.Scalar(arg1102_1, 0)
        new_zeros_default_156 = torch.ops.aten.new_zeros.default(add_tensor_164, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_156 = torch.ops.aten.where.self(le_scalar_103, new_zeros_default_156, add_tensor_164);  new_zeros_default_156 = add_tensor_164 = None
        slice_tensor_80 = torch.ops.aten.slice.Tensor(add_tensor_163, 1, 0, 336)
        slice_tensor_81 = torch.ops.aten.slice.Tensor(add_tensor_163, 1, 336, 672)
        slice_tensor_82 = torch.ops.aten.slice.Tensor(add_tensor_163, 1, 672, 1008)
        slice_tensor_83 = torch.ops.aten.slice.Tensor(add_tensor_163, 1, 1008, 1344);  add_tensor_163 = None
        max_pool2d_with_indices_backward_default_2 = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_tensor_83, arg1163_1, [3, 3], [2, 2], [0, 0], [1, 1], False, arg1164_1)
        constant_pad_nd_default_9 = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_default_2, [0, -1, 0, -1]);  max_pool2d_with_indices_backward_default_2 = None
        sum_dim_int_list_287 = torch.ops.aten.sum.dim_IntList(slice_tensor_83, [0, 2, 3])
        sub_tensor_504 = torch.ops.aten.sub.Tensor(arg1186_1, arg1882_1);  arg1186_1 = arg1882_1 = None
        mul_tensor_1451 = torch.ops.aten.mul.Tensor(slice_tensor_83, sub_tensor_504)
        sum_dim_int_list_288 = torch.ops.aten.sum.dim_IntList(mul_tensor_1451, [0, 2, 3]);  mul_tensor_1451 = None
        mul_tensor_1452 = torch.ops.aten.mul.Tensor(sum_dim_int_list_287, 0.0022675736961451248);  sum_dim_int_list_287 = None
        view_default_468 = torch.ops.aten.view.default(mul_tensor_1452, [1, 336, 1, 1]);  mul_tensor_1452 = None
        mul_tensor_1453 = torch.ops.aten.mul.Tensor(sum_dim_int_list_288, 0.0022675736961451248)
        mul_tensor_1454 = torch.ops.aten.mul.Tensor(arg1187_1, arg1187_1)
        mul_tensor_1455 = torch.ops.aten.mul.Tensor(mul_tensor_1453, mul_tensor_1454);  mul_tensor_1453 = mul_tensor_1454 = None
        view_default_469 = torch.ops.aten.view.default(mul_tensor_1455, [1, 336, 1, 1]);  mul_tensor_1455 = None
        mul_tensor_1456 = torch.ops.aten.mul.Tensor(arg1187_1, arg322_1);  arg322_1 = None
        view_default_470 = torch.ops.aten.view.default(mul_tensor_1456, [1, 336, 1, 1]);  mul_tensor_1456 = None
        mul_tensor_1457 = torch.ops.aten.mul.Tensor(sub_tensor_504, view_default_469);  sub_tensor_504 = view_default_469 = None
        sub_tensor_505 = torch.ops.aten.sub.Tensor(slice_tensor_83, mul_tensor_1457);  slice_tensor_83 = mul_tensor_1457 = None
        sub_tensor_506 = torch.ops.aten.sub.Tensor(sub_tensor_505, view_default_468);  sub_tensor_505 = view_default_468 = None
        mul_tensor_1458 = torch.ops.aten.mul.Tensor(sub_tensor_506, view_default_470);  sub_tensor_506 = view_default_470 = None
        mul_tensor_1459 = torch.ops.aten.mul.Tensor(sum_dim_int_list_288, arg1187_1);  sum_dim_int_list_288 = arg1187_1 = None
        convolution_backward_default_288 = torch.ops.aten.convolution_backward.default(mul_tensor_1458, arg1185_1, arg321_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1458 = arg1185_1 = arg321_1 = None
        getitem_864 = convolution_backward_default_288[0]
        getitem_865 = convolution_backward_default_288[1];  convolution_backward_default_288 = None
        convolution_backward_default_289 = torch.ops.aten.convolution_backward.default(getitem_864, arg1184_1, arg320_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_864 = arg320_1 = None
        getitem_867 = convolution_backward_default_289[0]
        getitem_868 = convolution_backward_default_289[1];  convolution_backward_default_289 = None
        le_scalar_104 = torch.ops.aten.le.Scalar(arg1184_1, 0);  arg1184_1 = None
        new_zeros_default_157 = torch.ops.aten.new_zeros.default(getitem_867, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_157 = torch.ops.aten.where.self(le_scalar_104, new_zeros_default_157, getitem_867);  le_scalar_104 = new_zeros_default_157 = getitem_867 = None
        sum_dim_int_list_289 = torch.ops.aten.sum.dim_IntList(where_self_157, [0, 2, 3])
        sub_tensor_507 = torch.ops.aten.sub.Tensor(arg1182_1, arg1883_1);  arg1182_1 = arg1883_1 = None
        mul_tensor_1460 = torch.ops.aten.mul.Tensor(where_self_157, sub_tensor_507)
        sum_dim_int_list_290 = torch.ops.aten.sum.dim_IntList(mul_tensor_1460, [0, 2, 3]);  mul_tensor_1460 = None
        mul_tensor_1461 = torch.ops.aten.mul.Tensor(sum_dim_int_list_289, 0.0022675736961451248);  sum_dim_int_list_289 = None
        view_default_471 = torch.ops.aten.view.default(mul_tensor_1461, [1, 336, 1, 1]);  mul_tensor_1461 = None
        mul_tensor_1462 = torch.ops.aten.mul.Tensor(sum_dim_int_list_290, 0.0022675736961451248)
        mul_tensor_1463 = torch.ops.aten.mul.Tensor(arg1183_1, arg1183_1)
        mul_tensor_1464 = torch.ops.aten.mul.Tensor(mul_tensor_1462, mul_tensor_1463);  mul_tensor_1462 = mul_tensor_1463 = None
        view_default_472 = torch.ops.aten.view.default(mul_tensor_1464, [1, 336, 1, 1]);  mul_tensor_1464 = None
        mul_tensor_1465 = torch.ops.aten.mul.Tensor(arg1183_1, arg319_1);  arg319_1 = None
        view_default_473 = torch.ops.aten.view.default(mul_tensor_1465, [1, 336, 1, 1]);  mul_tensor_1465 = None
        mul_tensor_1466 = torch.ops.aten.mul.Tensor(sub_tensor_507, view_default_472);  sub_tensor_507 = view_default_472 = None
        sub_tensor_508 = torch.ops.aten.sub.Tensor(where_self_157, mul_tensor_1466);  where_self_157 = mul_tensor_1466 = None
        sub_tensor_509 = torch.ops.aten.sub.Tensor(sub_tensor_508, view_default_471);  sub_tensor_508 = view_default_471 = None
        mul_tensor_1467 = torch.ops.aten.mul.Tensor(sub_tensor_509, view_default_473);  sub_tensor_509 = view_default_473 = None
        mul_tensor_1468 = torch.ops.aten.mul.Tensor(sum_dim_int_list_290, arg1183_1);  sum_dim_int_list_290 = arg1183_1 = None
        convolution_backward_default_290 = torch.ops.aten.convolution_backward.default(mul_tensor_1467, arg1181_1, arg318_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1467 = arg1181_1 = arg318_1 = None
        getitem_870 = convolution_backward_default_290[0]
        getitem_871 = convolution_backward_default_290[1];  convolution_backward_default_290 = None
        convolution_backward_default_291 = torch.ops.aten.convolution_backward.default(getitem_870, relu_default_14, arg317_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_870 = arg317_1 = None
        getitem_873 = convolution_backward_default_291[0]
        getitem_874 = convolution_backward_default_291[1];  convolution_backward_default_291 = None
        le_scalar_105 = torch.ops.aten.le.Scalar(relu_default_14, 0);  relu_default_14 = None
        new_zeros_default_158 = torch.ops.aten.new_zeros.default(getitem_873, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_158 = torch.ops.aten.where.self(le_scalar_105, new_zeros_default_158, getitem_873);  le_scalar_105 = new_zeros_default_158 = getitem_873 = None
        add_tensor_165 = torch.ops.aten.add.Tensor(slice_tensor_80, slice_tensor_82);  slice_tensor_80 = None
        avg_pool2d_backward_default_30 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_82, arg1162_1, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_82 = arg1162_1 = None
        add_tensor_166 = torch.ops.aten.add.Tensor(where_self_158, avg_pool2d_backward_default_30);  where_self_158 = avg_pool2d_backward_default_30 = None
        sum_dim_int_list_291 = torch.ops.aten.sum.dim_IntList(slice_tensor_81, [0, 2, 3])
        sub_tensor_510 = torch.ops.aten.sub.Tensor(arg1179_1, arg1884_1);  arg1179_1 = arg1884_1 = None
        mul_tensor_1469 = torch.ops.aten.mul.Tensor(slice_tensor_81, sub_tensor_510)
        sum_dim_int_list_292 = torch.ops.aten.sum.dim_IntList(mul_tensor_1469, [0, 2, 3]);  mul_tensor_1469 = None
        mul_tensor_1470 = torch.ops.aten.mul.Tensor(sum_dim_int_list_291, 0.0022675736961451248);  sum_dim_int_list_291 = None
        view_default_474 = torch.ops.aten.view.default(mul_tensor_1470, [1, 336, 1, 1]);  mul_tensor_1470 = None
        mul_tensor_1471 = torch.ops.aten.mul.Tensor(sum_dim_int_list_292, 0.0022675736961451248)
        mul_tensor_1472 = torch.ops.aten.mul.Tensor(arg1180_1, arg1180_1)
        mul_tensor_1473 = torch.ops.aten.mul.Tensor(mul_tensor_1471, mul_tensor_1472);  mul_tensor_1471 = mul_tensor_1472 = None
        view_default_475 = torch.ops.aten.view.default(mul_tensor_1473, [1, 336, 1, 1]);  mul_tensor_1473 = None
        mul_tensor_1474 = torch.ops.aten.mul.Tensor(arg1180_1, arg316_1);  arg316_1 = None
        view_default_476 = torch.ops.aten.view.default(mul_tensor_1474, [1, 336, 1, 1]);  mul_tensor_1474 = None
        mul_tensor_1475 = torch.ops.aten.mul.Tensor(sub_tensor_510, view_default_475);  sub_tensor_510 = view_default_475 = None
        sub_tensor_511 = torch.ops.aten.sub.Tensor(slice_tensor_81, mul_tensor_1475);  mul_tensor_1475 = None
        sub_tensor_512 = torch.ops.aten.sub.Tensor(sub_tensor_511, view_default_474);  sub_tensor_511 = view_default_474 = None
        mul_tensor_1476 = torch.ops.aten.mul.Tensor(sub_tensor_512, view_default_476);  sub_tensor_512 = view_default_476 = None
        mul_tensor_1477 = torch.ops.aten.mul.Tensor(sum_dim_int_list_292, arg1180_1);  sum_dim_int_list_292 = arg1180_1 = None
        convolution_backward_default_292 = torch.ops.aten.convolution_backward.default(mul_tensor_1476, arg1178_1, arg315_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1476 = arg1178_1 = arg315_1 = None
        getitem_876 = convolution_backward_default_292[0]
        getitem_877 = convolution_backward_default_292[1];  convolution_backward_default_292 = None
        convolution_backward_default_293 = torch.ops.aten.convolution_backward.default(getitem_876, arg1177_1, arg314_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_876 = arg314_1 = None
        getitem_879 = convolution_backward_default_293[0]
        getitem_880 = convolution_backward_default_293[1];  convolution_backward_default_293 = None
        le_scalar_106 = torch.ops.aten.le.Scalar(arg1177_1, 0);  arg1177_1 = None
        new_zeros_default_159 = torch.ops.aten.new_zeros.default(getitem_879, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_159 = torch.ops.aten.where.self(le_scalar_106, new_zeros_default_159, getitem_879);  le_scalar_106 = new_zeros_default_159 = getitem_879 = None
        sum_dim_int_list_293 = torch.ops.aten.sum.dim_IntList(where_self_159, [0, 2, 3])
        sub_tensor_513 = torch.ops.aten.sub.Tensor(arg1175_1, arg1885_1);  arg1175_1 = arg1885_1 = None
        mul_tensor_1478 = torch.ops.aten.mul.Tensor(where_self_159, sub_tensor_513)
        sum_dim_int_list_294 = torch.ops.aten.sum.dim_IntList(mul_tensor_1478, [0, 2, 3]);  mul_tensor_1478 = None
        mul_tensor_1479 = torch.ops.aten.mul.Tensor(sum_dim_int_list_293, 0.0022675736961451248);  sum_dim_int_list_293 = None
        view_default_477 = torch.ops.aten.view.default(mul_tensor_1479, [1, 336, 1, 1]);  mul_tensor_1479 = None
        mul_tensor_1480 = torch.ops.aten.mul.Tensor(sum_dim_int_list_294, 0.0022675736961451248)
        mul_tensor_1481 = torch.ops.aten.mul.Tensor(arg1176_1, arg1176_1)
        mul_tensor_1482 = torch.ops.aten.mul.Tensor(mul_tensor_1480, mul_tensor_1481);  mul_tensor_1480 = mul_tensor_1481 = None
        view_default_478 = torch.ops.aten.view.default(mul_tensor_1482, [1, 336, 1, 1]);  mul_tensor_1482 = None
        mul_tensor_1483 = torch.ops.aten.mul.Tensor(arg1176_1, arg313_1);  arg313_1 = None
        view_default_479 = torch.ops.aten.view.default(mul_tensor_1483, [1, 336, 1, 1]);  mul_tensor_1483 = None
        mul_tensor_1484 = torch.ops.aten.mul.Tensor(sub_tensor_513, view_default_478);  sub_tensor_513 = view_default_478 = None
        sub_tensor_514 = torch.ops.aten.sub.Tensor(where_self_159, mul_tensor_1484);  where_self_159 = mul_tensor_1484 = None
        sub_tensor_515 = torch.ops.aten.sub.Tensor(sub_tensor_514, view_default_477);  sub_tensor_514 = view_default_477 = None
        mul_tensor_1485 = torch.ops.aten.mul.Tensor(sub_tensor_515, view_default_479);  sub_tensor_515 = view_default_479 = None
        mul_tensor_1486 = torch.ops.aten.mul.Tensor(sum_dim_int_list_294, arg1176_1);  sum_dim_int_list_294 = arg1176_1 = None
        convolution_backward_default_294 = torch.ops.aten.convolution_backward.default(mul_tensor_1485, arg1174_1, arg312_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1485 = arg1174_1 = arg312_1 = None
        getitem_882 = convolution_backward_default_294[0]
        getitem_883 = convolution_backward_default_294[1];  convolution_backward_default_294 = None
        convolution_backward_default_295 = torch.ops.aten.convolution_backward.default(getitem_882, arg1173_1, arg11_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_882 = arg1173_1 = arg11_1 = None
        getitem_885 = convolution_backward_default_295[0]
        getitem_886 = convolution_backward_default_295[1];  convolution_backward_default_295 = None
        constant_pad_nd_default_10 = torch.ops.aten.constant_pad_nd.default(getitem_885, [-1, -2, -1, -2]);  getitem_885 = None
        new_zeros_default_160 = torch.ops.aten.new_zeros.default(constant_pad_nd_default_10, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_160 = torch.ops.aten.where.self(arg1886_1, new_zeros_default_160, constant_pad_nd_default_10);  new_zeros_default_160 = constant_pad_nd_default_10 = None
        avg_pool2d_backward_default_31 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_81, arg1172_1, [3, 3], [2, 2], [0, 0], False, False, None);  slice_tensor_81 = arg1172_1 = None
        constant_pad_nd_default_11 = torch.ops.aten.constant_pad_nd.default(avg_pool2d_backward_default_31, [0, -1, 0, -1]);  avg_pool2d_backward_default_31 = None
        add_tensor_167 = torch.ops.aten.add.Tensor(constant_pad_nd_default_9, constant_pad_nd_default_11);  constant_pad_nd_default_9 = constant_pad_nd_default_11 = None
        sum_dim_int_list_295 = torch.ops.aten.sum.dim_IntList(add_tensor_165, [0, 2, 3])
        sub_tensor_516 = torch.ops.aten.sub.Tensor(arg1170_1, arg1887_1);  arg1170_1 = arg1887_1 = None
        mul_tensor_1487 = torch.ops.aten.mul.Tensor(add_tensor_165, sub_tensor_516)
        sum_dim_int_list_296 = torch.ops.aten.sum.dim_IntList(mul_tensor_1487, [0, 2, 3]);  mul_tensor_1487 = None
        mul_tensor_1488 = torch.ops.aten.mul.Tensor(sum_dim_int_list_295, 0.0022675736961451248);  sum_dim_int_list_295 = None
        view_default_480 = torch.ops.aten.view.default(mul_tensor_1488, [1, 336, 1, 1]);  mul_tensor_1488 = None
        mul_tensor_1489 = torch.ops.aten.mul.Tensor(sum_dim_int_list_296, 0.0022675736961451248)
        mul_tensor_1490 = torch.ops.aten.mul.Tensor(arg1171_1, arg1171_1)
        mul_tensor_1491 = torch.ops.aten.mul.Tensor(mul_tensor_1489, mul_tensor_1490);  mul_tensor_1489 = mul_tensor_1490 = None
        view_default_481 = torch.ops.aten.view.default(mul_tensor_1491, [1, 336, 1, 1]);  mul_tensor_1491 = None
        mul_tensor_1492 = torch.ops.aten.mul.Tensor(arg1171_1, arg311_1);  arg311_1 = None
        view_default_482 = torch.ops.aten.view.default(mul_tensor_1492, [1, 336, 1, 1]);  mul_tensor_1492 = None
        mul_tensor_1493 = torch.ops.aten.mul.Tensor(sub_tensor_516, view_default_481);  sub_tensor_516 = view_default_481 = None
        sub_tensor_517 = torch.ops.aten.sub.Tensor(add_tensor_165, mul_tensor_1493);  mul_tensor_1493 = None
        sub_tensor_518 = torch.ops.aten.sub.Tensor(sub_tensor_517, view_default_480);  sub_tensor_517 = view_default_480 = None
        mul_tensor_1494 = torch.ops.aten.mul.Tensor(sub_tensor_518, view_default_482);  sub_tensor_518 = view_default_482 = None
        mul_tensor_1495 = torch.ops.aten.mul.Tensor(sum_dim_int_list_296, arg1171_1);  sum_dim_int_list_296 = arg1171_1 = None
        convolution_backward_default_296 = torch.ops.aten.convolution_backward.default(mul_tensor_1494, arg1169_1, arg310_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1494 = arg1169_1 = arg310_1 = None
        getitem_888 = convolution_backward_default_296[0]
        getitem_889 = convolution_backward_default_296[1];  convolution_backward_default_296 = None
        convolution_backward_default_297 = torch.ops.aten.convolution_backward.default(getitem_888, arg1168_1, arg309_1, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_888 = arg309_1 = None
        getitem_891 = convolution_backward_default_297[0]
        getitem_892 = convolution_backward_default_297[1];  convolution_backward_default_297 = None
        le_scalar_107 = torch.ops.aten.le.Scalar(arg1168_1, 0);  arg1168_1 = None
        new_zeros_default_161 = torch.ops.aten.new_zeros.default(getitem_891, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_161 = torch.ops.aten.where.self(le_scalar_107, new_zeros_default_161, getitem_891);  le_scalar_107 = new_zeros_default_161 = getitem_891 = None
        sum_dim_int_list_297 = torch.ops.aten.sum.dim_IntList(where_self_161, [0, 2, 3])
        sub_tensor_519 = torch.ops.aten.sub.Tensor(arg1166_1, arg1888_1);  arg1166_1 = arg1888_1 = None
        mul_tensor_1496 = torch.ops.aten.mul.Tensor(where_self_161, sub_tensor_519)
        sum_dim_int_list_298 = torch.ops.aten.sum.dim_IntList(mul_tensor_1496, [0, 2, 3]);  mul_tensor_1496 = None
        mul_tensor_1497 = torch.ops.aten.mul.Tensor(sum_dim_int_list_297, 0.0022675736961451248);  sum_dim_int_list_297 = None
        view_default_483 = torch.ops.aten.view.default(mul_tensor_1497, [1, 336, 1, 1]);  mul_tensor_1497 = None
        mul_tensor_1498 = torch.ops.aten.mul.Tensor(sum_dim_int_list_298, 0.0022675736961451248)
        mul_tensor_1499 = torch.ops.aten.mul.Tensor(arg1167_1, arg1167_1)
        mul_tensor_1500 = torch.ops.aten.mul.Tensor(mul_tensor_1498, mul_tensor_1499);  mul_tensor_1498 = mul_tensor_1499 = None
        view_default_484 = torch.ops.aten.view.default(mul_tensor_1500, [1, 336, 1, 1]);  mul_tensor_1500 = None
        mul_tensor_1501 = torch.ops.aten.mul.Tensor(arg1167_1, arg308_1);  arg308_1 = None
        view_default_485 = torch.ops.aten.view.default(mul_tensor_1501, [1, 336, 1, 1]);  mul_tensor_1501 = None
        mul_tensor_1502 = torch.ops.aten.mul.Tensor(sub_tensor_519, view_default_484);  sub_tensor_519 = view_default_484 = None
        sub_tensor_520 = torch.ops.aten.sub.Tensor(where_self_161, mul_tensor_1502);  where_self_161 = mul_tensor_1502 = None
        sub_tensor_521 = torch.ops.aten.sub.Tensor(sub_tensor_520, view_default_483);  sub_tensor_520 = view_default_483 = None
        mul_tensor_1503 = torch.ops.aten.mul.Tensor(sub_tensor_521, view_default_485);  sub_tensor_521 = view_default_485 = None
        mul_tensor_1504 = torch.ops.aten.mul.Tensor(sum_dim_int_list_298, arg1167_1);  sum_dim_int_list_298 = arg1167_1 = None
        convolution_backward_default_298 = torch.ops.aten.convolution_backward.default(mul_tensor_1503, arg1165_1, arg307_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1503 = arg1165_1 = arg307_1 = None
        getitem_894 = convolution_backward_default_298[0]
        getitem_895 = convolution_backward_default_298[1];  convolution_backward_default_298 = None
        convolution_backward_default_299 = torch.ops.aten.convolution_backward.default(getitem_894, arg1154_1, arg10_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_894 = arg10_1 = None
        getitem_897 = convolution_backward_default_299[0]
        getitem_898 = convolution_backward_default_299[1];  convolution_backward_default_299 = None
        constant_pad_nd_default_12 = torch.ops.aten.constant_pad_nd.default(getitem_897, [-2, -3, -2, -3]);  getitem_897 = None
        new_zeros_default_162 = torch.ops.aten.new_zeros.default(constant_pad_nd_default_12, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_162 = torch.ops.aten.where.self(arg1886_1, new_zeros_default_162, constant_pad_nd_default_12);  new_zeros_default_162 = constant_pad_nd_default_12 = None
        add_tensor_168 = torch.ops.aten.add.Tensor(where_self_160, where_self_162);  where_self_160 = where_self_162 = None
        max_pool2d_with_indices_backward_default_3 = torch.ops.aten.max_pool2d_with_indices_backward.default(add_tensor_165, arg1163_1, [3, 3], [2, 2], [0, 0], [1, 1], False, arg1164_1);  add_tensor_165 = arg1163_1 = arg1164_1 = None
        constant_pad_nd_default_13 = torch.ops.aten.constant_pad_nd.default(max_pool2d_with_indices_backward_default_3, [0, -1, 0, -1]);  max_pool2d_with_indices_backward_default_3 = None
        add_tensor_169 = torch.ops.aten.add.Tensor(add_tensor_167, constant_pad_nd_default_13);  add_tensor_167 = constant_pad_nd_default_13 = None
        sum_dim_int_list_299 = torch.ops.aten.sum.dim_IntList(add_tensor_166, [0, 2, 3])
        sub_tensor_522 = torch.ops.aten.sub.Tensor(arg1160_1, arg1889_1);  arg1160_1 = arg1889_1 = None
        mul_tensor_1505 = torch.ops.aten.mul.Tensor(add_tensor_166, sub_tensor_522)
        sum_dim_int_list_300 = torch.ops.aten.sum.dim_IntList(mul_tensor_1505, [0, 2, 3]);  mul_tensor_1505 = None
        mul_tensor_1506 = torch.ops.aten.mul.Tensor(sum_dim_int_list_299, 0.0022675736961451248);  sum_dim_int_list_299 = None
        view_default_486 = torch.ops.aten.view.default(mul_tensor_1506, [1, 336, 1, 1]);  mul_tensor_1506 = None
        mul_tensor_1507 = torch.ops.aten.mul.Tensor(sum_dim_int_list_300, 0.0022675736961451248)
        mul_tensor_1508 = torch.ops.aten.mul.Tensor(arg1161_1, arg1161_1)
        mul_tensor_1509 = torch.ops.aten.mul.Tensor(mul_tensor_1507, mul_tensor_1508);  mul_tensor_1507 = mul_tensor_1508 = None
        view_default_487 = torch.ops.aten.view.default(mul_tensor_1509, [1, 336, 1, 1]);  mul_tensor_1509 = None
        mul_tensor_1510 = torch.ops.aten.mul.Tensor(arg1161_1, arg306_1);  arg306_1 = None
        view_default_488 = torch.ops.aten.view.default(mul_tensor_1510, [1, 336, 1, 1]);  mul_tensor_1510 = None
        mul_tensor_1511 = torch.ops.aten.mul.Tensor(sub_tensor_522, view_default_487);  sub_tensor_522 = view_default_487 = None
        sub_tensor_523 = torch.ops.aten.sub.Tensor(add_tensor_166, mul_tensor_1511);  mul_tensor_1511 = None
        sub_tensor_524 = torch.ops.aten.sub.Tensor(sub_tensor_523, view_default_486);  sub_tensor_523 = None
        mul_tensor_1512 = torch.ops.aten.mul.Tensor(sub_tensor_524, view_default_488);  sub_tensor_524 = view_default_488 = None
        mul_tensor_1513 = torch.ops.aten.mul.Tensor(sum_dim_int_list_300, arg1161_1);  sum_dim_int_list_300 = arg1161_1 = None
        convolution_backward_default_300 = torch.ops.aten.convolution_backward.default(mul_tensor_1512, arg1159_1, arg305_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1512 = arg1159_1 = arg305_1 = None
        getitem_900 = convolution_backward_default_300[0]
        getitem_901 = convolution_backward_default_300[1];  convolution_backward_default_300 = None
        convolution_backward_default_301 = torch.ops.aten.convolution_backward.default(getitem_900, arg1158_1, arg304_1, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_900 = arg304_1 = None
        getitem_903 = convolution_backward_default_301[0]
        getitem_904 = convolution_backward_default_301[1];  convolution_backward_default_301 = None
        le_scalar_108 = torch.ops.aten.le.Scalar(arg1158_1, 0);  arg1158_1 = None
        new_zeros_default_163 = torch.ops.aten.new_zeros.default(getitem_903, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_163 = torch.ops.aten.where.self(le_scalar_108, new_zeros_default_163, getitem_903);  le_scalar_108 = new_zeros_default_163 = getitem_903 = None
        sum_dim_int_list_301 = torch.ops.aten.sum.dim_IntList(where_self_163, [0, 2, 3])
        sub_tensor_525 = torch.ops.aten.sub.Tensor(arg1156_1, arg1890_1);  arg1156_1 = arg1890_1 = None
        mul_tensor_1514 = torch.ops.aten.mul.Tensor(where_self_163, sub_tensor_525)
        sum_dim_int_list_302 = torch.ops.aten.sum.dim_IntList(mul_tensor_1514, [0, 2, 3]);  mul_tensor_1514 = None
        mul_tensor_1515 = torch.ops.aten.mul.Tensor(sum_dim_int_list_301, 0.0022675736961451248);  sum_dim_int_list_301 = None
        view_default_489 = torch.ops.aten.view.default(mul_tensor_1515, [1, 336, 1, 1]);  mul_tensor_1515 = None
        mul_tensor_1516 = torch.ops.aten.mul.Tensor(sum_dim_int_list_302, 0.0022675736961451248)
        mul_tensor_1517 = torch.ops.aten.mul.Tensor(arg1157_1, arg1157_1)
        mul_tensor_1518 = torch.ops.aten.mul.Tensor(mul_tensor_1516, mul_tensor_1517);  mul_tensor_1516 = mul_tensor_1517 = None
        view_default_490 = torch.ops.aten.view.default(mul_tensor_1518, [1, 336, 1, 1]);  mul_tensor_1518 = None
        mul_tensor_1519 = torch.ops.aten.mul.Tensor(arg1157_1, arg303_1);  arg303_1 = None
        view_default_491 = torch.ops.aten.view.default(mul_tensor_1519, [1, 336, 1, 1]);  mul_tensor_1519 = None
        mul_tensor_1520 = torch.ops.aten.mul.Tensor(sub_tensor_525, view_default_490);  sub_tensor_525 = view_default_490 = None
        sub_tensor_526 = torch.ops.aten.sub.Tensor(where_self_163, mul_tensor_1520);  where_self_163 = mul_tensor_1520 = None
        sub_tensor_527 = torch.ops.aten.sub.Tensor(sub_tensor_526, view_default_489);  sub_tensor_526 = view_default_489 = None
        mul_tensor_1521 = torch.ops.aten.mul.Tensor(sub_tensor_527, view_default_491);  sub_tensor_527 = view_default_491 = None
        mul_tensor_1522 = torch.ops.aten.mul.Tensor(sum_dim_int_list_302, arg1157_1);  sum_dim_int_list_302 = arg1157_1 = None
        convolution_backward_default_302 = torch.ops.aten.convolution_backward.default(mul_tensor_1521, arg1155_1, arg302_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1521 = arg1155_1 = arg302_1 = None
        getitem_906 = convolution_backward_default_302[0]
        getitem_907 = convolution_backward_default_302[1];  convolution_backward_default_302 = None
        convolution_backward_default_303 = torch.ops.aten.convolution_backward.default(getitem_906, arg1154_1, arg9_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_906 = arg1154_1 = arg9_1 = None
        getitem_909 = convolution_backward_default_303[0]
        getitem_910 = convolution_backward_default_303[1];  convolution_backward_default_303 = None
        constant_pad_nd_default_14 = torch.ops.aten.constant_pad_nd.default(getitem_909, [-2, -3, -2, -3]);  getitem_909 = None
        new_zeros_default_164 = torch.ops.aten.new_zeros.default(constant_pad_nd_default_14, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_164 = torch.ops.aten.where.self(arg1886_1, new_zeros_default_164, constant_pad_nd_default_14);  arg1886_1 = new_zeros_default_164 = constant_pad_nd_default_14 = None
        add_tensor_170 = torch.ops.aten.add.Tensor(add_tensor_168, where_self_164);  add_tensor_168 = where_self_164 = None
        sub_tensor_528 = torch.ops.aten.sub.Tensor(arg1152_1, arg1891_1);  arg1152_1 = arg1891_1 = None
        mul_tensor_1523 = torch.ops.aten.mul.Tensor(add_tensor_166, sub_tensor_528)
        sum_dim_int_list_303 = torch.ops.aten.sum.dim_IntList(mul_tensor_1523, [0, 2, 3]);  mul_tensor_1523 = None
        mul_tensor_1524 = torch.ops.aten.mul.Tensor(sum_dim_int_list_303, 0.0022675736961451248)
        mul_tensor_1525 = torch.ops.aten.mul.Tensor(arg1153_1, arg1153_1)
        mul_tensor_1526 = torch.ops.aten.mul.Tensor(mul_tensor_1524, mul_tensor_1525);  mul_tensor_1524 = mul_tensor_1525 = None
        view_default_492 = torch.ops.aten.view.default(mul_tensor_1526, [1, 336, 1, 1]);  mul_tensor_1526 = None
        mul_tensor_1527 = torch.ops.aten.mul.Tensor(arg1153_1, arg301_1);  arg301_1 = None
        view_default_493 = torch.ops.aten.view.default(mul_tensor_1527, [1, 336, 1, 1]);  mul_tensor_1527 = None
        mul_tensor_1528 = torch.ops.aten.mul.Tensor(sub_tensor_528, view_default_492);  sub_tensor_528 = view_default_492 = None
        sub_tensor_529 = torch.ops.aten.sub.Tensor(add_tensor_166, mul_tensor_1528);  add_tensor_166 = mul_tensor_1528 = None
        sub_tensor_530 = torch.ops.aten.sub.Tensor(sub_tensor_529, view_default_486);  sub_tensor_529 = view_default_486 = None
        mul_tensor_1529 = torch.ops.aten.mul.Tensor(sub_tensor_530, view_default_493);  sub_tensor_530 = view_default_493 = None
        mul_tensor_1530 = torch.ops.aten.mul.Tensor(sum_dim_int_list_303, arg1153_1);  sum_dim_int_list_303 = arg1153_1 = None
        convolution_backward_default_304 = torch.ops.aten.convolution_backward.default(mul_tensor_1529, arg1151_1, arg300_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1529 = arg1151_1 = arg300_1 = None
        getitem_912 = convolution_backward_default_304[0]
        getitem_913 = convolution_backward_default_304[1];  convolution_backward_default_304 = None
        convolution_backward_default_305 = torch.ops.aten.convolution_backward.default(getitem_912, arg1150_1, arg299_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_912 = arg299_1 = None
        getitem_915 = convolution_backward_default_305[0]
        getitem_916 = convolution_backward_default_305[1];  convolution_backward_default_305 = None
        le_scalar_109 = torch.ops.aten.le.Scalar(arg1150_1, 0);  arg1150_1 = None
        new_zeros_default_165 = torch.ops.aten.new_zeros.default(getitem_915, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_165 = torch.ops.aten.where.self(le_scalar_109, new_zeros_default_165, getitem_915);  le_scalar_109 = new_zeros_default_165 = getitem_915 = None
        sum_dim_int_list_304 = torch.ops.aten.sum.dim_IntList(where_self_165, [0, 2, 3])
        sub_tensor_531 = torch.ops.aten.sub.Tensor(arg1148_1, arg1892_1);  arg1148_1 = arg1892_1 = None
        mul_tensor_1531 = torch.ops.aten.mul.Tensor(where_self_165, sub_tensor_531)
        sum_dim_int_list_305 = torch.ops.aten.sum.dim_IntList(mul_tensor_1531, [0, 2, 3]);  mul_tensor_1531 = None
        mul_tensor_1532 = torch.ops.aten.mul.Tensor(sum_dim_int_list_304, 0.0022675736961451248);  sum_dim_int_list_304 = None
        view_default_494 = torch.ops.aten.view.default(mul_tensor_1532, [1, 336, 1, 1]);  mul_tensor_1532 = None
        mul_tensor_1533 = torch.ops.aten.mul.Tensor(sum_dim_int_list_305, 0.0022675736961451248)
        mul_tensor_1534 = torch.ops.aten.mul.Tensor(arg1149_1, arg1149_1)
        mul_tensor_1535 = torch.ops.aten.mul.Tensor(mul_tensor_1533, mul_tensor_1534);  mul_tensor_1533 = mul_tensor_1534 = None
        view_default_495 = torch.ops.aten.view.default(mul_tensor_1535, [1, 336, 1, 1]);  mul_tensor_1535 = None
        mul_tensor_1536 = torch.ops.aten.mul.Tensor(arg1149_1, arg298_1);  arg298_1 = None
        view_default_496 = torch.ops.aten.view.default(mul_tensor_1536, [1, 336, 1, 1]);  mul_tensor_1536 = None
        mul_tensor_1537 = torch.ops.aten.mul.Tensor(sub_tensor_531, view_default_495);  sub_tensor_531 = view_default_495 = None
        sub_tensor_532 = torch.ops.aten.sub.Tensor(where_self_165, mul_tensor_1537);  where_self_165 = mul_tensor_1537 = None
        sub_tensor_533 = torch.ops.aten.sub.Tensor(sub_tensor_532, view_default_494);  sub_tensor_532 = view_default_494 = None
        mul_tensor_1538 = torch.ops.aten.mul.Tensor(sub_tensor_533, view_default_496);  sub_tensor_533 = view_default_496 = None
        mul_tensor_1539 = torch.ops.aten.mul.Tensor(sum_dim_int_list_305, arg1149_1);  sum_dim_int_list_305 = arg1149_1 = None
        convolution_backward_default_306 = torch.ops.aten.convolution_backward.default(mul_tensor_1538, arg1147_1, arg297_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1538 = arg1147_1 = arg297_1 = None
        getitem_918 = convolution_backward_default_306[0]
        getitem_919 = convolution_backward_default_306[1];  convolution_backward_default_306 = None
        convolution_backward_default_307 = torch.ops.aten.convolution_backward.default(getitem_918, arg1146_1, arg8_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 336, [True, True, False]);  getitem_918 = arg1146_1 = arg8_1 = None
        getitem_921 = convolution_backward_default_307[0]
        getitem_922 = convolution_backward_default_307[1];  convolution_backward_default_307 = None
        constant_pad_nd_default_15 = torch.ops.aten.constant_pad_nd.default(getitem_921, [-1, -2, -1, -2]);  getitem_921 = None
        new_zeros_default_166 = torch.ops.aten.new_zeros.default(constant_pad_nd_default_15, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_166 = torch.ops.aten.where.self(arg1893_1, new_zeros_default_166, constant_pad_nd_default_15);  arg1893_1 = new_zeros_default_166 = constant_pad_nd_default_15 = None
        add_tensor_171 = torch.ops.aten.add.Tensor(add_tensor_169, where_self_166);  add_tensor_169 = where_self_166 = None
        sum_dim_int_list_306 = torch.ops.aten.sum.dim_IntList(add_tensor_171, [0, 2, 3])
        sub_tensor_534 = torch.ops.aten.sub.Tensor(arg1144_1, arg1894_1);  arg1144_1 = arg1894_1 = None
        mul_tensor_1540 = torch.ops.aten.mul.Tensor(add_tensor_171, sub_tensor_534)
        sum_dim_int_list_307 = torch.ops.aten.sum.dim_IntList(mul_tensor_1540, [0, 2, 3]);  mul_tensor_1540 = None
        mul_tensor_1541 = torch.ops.aten.mul.Tensor(sum_dim_int_list_306, 0.0005668934240362812);  sum_dim_int_list_306 = None
        view_default_497 = torch.ops.aten.view.default(mul_tensor_1541, [1, 336, 1, 1]);  mul_tensor_1541 = None
        mul_tensor_1542 = torch.ops.aten.mul.Tensor(sum_dim_int_list_307, 0.0005668934240362812)
        mul_tensor_1543 = torch.ops.aten.mul.Tensor(arg1145_1, arg1145_1)
        mul_tensor_1544 = torch.ops.aten.mul.Tensor(mul_tensor_1542, mul_tensor_1543);  mul_tensor_1542 = mul_tensor_1543 = None
        view_default_498 = torch.ops.aten.view.default(mul_tensor_1544, [1, 336, 1, 1]);  mul_tensor_1544 = None
        mul_tensor_1545 = torch.ops.aten.mul.Tensor(arg1145_1, arg296_1);  arg296_1 = None
        view_default_499 = torch.ops.aten.view.default(mul_tensor_1545, [1, 336, 1, 1]);  mul_tensor_1545 = None
        mul_tensor_1546 = torch.ops.aten.mul.Tensor(sub_tensor_534, view_default_498);  sub_tensor_534 = view_default_498 = None
        sub_tensor_535 = torch.ops.aten.sub.Tensor(add_tensor_171, mul_tensor_1546);  add_tensor_171 = mul_tensor_1546 = None
        sub_tensor_536 = torch.ops.aten.sub.Tensor(sub_tensor_535, view_default_497);  sub_tensor_535 = view_default_497 = None
        mul_tensor_1547 = torch.ops.aten.mul.Tensor(sub_tensor_536, view_default_499);  sub_tensor_536 = view_default_499 = None
        mul_tensor_1548 = torch.ops.aten.mul.Tensor(sum_dim_int_list_307, arg1145_1);  sum_dim_int_list_307 = arg1145_1 = None
        convolution_backward_default_308 = torch.ops.aten.convolution_backward.default(mul_tensor_1547, arg1143_1, arg295_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1547 = arg295_1 = None
        getitem_924 = convolution_backward_default_308[0]
        getitem_925 = convolution_backward_default_308[1];  convolution_backward_default_308 = None
        le_scalar_110 = torch.ops.aten.le.Scalar(arg1143_1, 0);  arg1143_1 = None
        new_zeros_default_167 = torch.ops.aten.new_zeros.default(getitem_924, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_167 = torch.ops.aten.where.self(le_scalar_110, new_zeros_default_167, getitem_924);  le_scalar_110 = new_zeros_default_167 = getitem_924 = None
        sum_dim_int_list_308 = torch.ops.aten.sum.dim_IntList(add_tensor_170, [0, 2, 3])
        sub_tensor_537 = torch.ops.aten.sub.Tensor(arg1141_1, arg1895_1);  arg1141_1 = arg1895_1 = None
        mul_tensor_1549 = torch.ops.aten.mul.Tensor(add_tensor_170, sub_tensor_537)
        sum_dim_int_list_309 = torch.ops.aten.sum.dim_IntList(mul_tensor_1549, [0, 2, 3]);  mul_tensor_1549 = None
        mul_tensor_1550 = torch.ops.aten.mul.Tensor(sum_dim_int_list_308, 0.0005668934240362812);  sum_dim_int_list_308 = None
        view_default_500 = torch.ops.aten.view.default(mul_tensor_1550, [1, 336, 1, 1]);  mul_tensor_1550 = None
        mul_tensor_1551 = torch.ops.aten.mul.Tensor(sum_dim_int_list_309, 0.0005668934240362812)
        mul_tensor_1552 = torch.ops.aten.mul.Tensor(arg1142_1, arg1142_1)
        mul_tensor_1553 = torch.ops.aten.mul.Tensor(mul_tensor_1551, mul_tensor_1552);  mul_tensor_1551 = mul_tensor_1552 = None
        view_default_501 = torch.ops.aten.view.default(mul_tensor_1553, [1, 336, 1, 1]);  mul_tensor_1553 = None
        mul_tensor_1554 = torch.ops.aten.mul.Tensor(arg1142_1, arg294_1);  arg294_1 = None
        view_default_502 = torch.ops.aten.view.default(mul_tensor_1554, [1, 336, 1, 1]);  mul_tensor_1554 = None
        mul_tensor_1555 = torch.ops.aten.mul.Tensor(sub_tensor_537, view_default_501);  sub_tensor_537 = view_default_501 = None
        sub_tensor_538 = torch.ops.aten.sub.Tensor(add_tensor_170, mul_tensor_1555);  add_tensor_170 = mul_tensor_1555 = None
        sub_tensor_539 = torch.ops.aten.sub.Tensor(sub_tensor_538, view_default_500);  sub_tensor_538 = view_default_500 = None
        mul_tensor_1556 = torch.ops.aten.mul.Tensor(sub_tensor_539, view_default_502);  sub_tensor_539 = view_default_502 = None
        mul_tensor_1557 = torch.ops.aten.mul.Tensor(sum_dim_int_list_309, arg1142_1);  sum_dim_int_list_309 = arg1142_1 = None
        convolution_backward_default_309 = torch.ops.aten.convolution_backward.default(mul_tensor_1556, arg1102_1, arg293_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1556 = arg293_1 = None
        getitem_927 = convolution_backward_default_309[0]
        getitem_928 = convolution_backward_default_309[1];  convolution_backward_default_309 = None
        new_zeros_default_168 = torch.ops.aten.new_zeros.default(getitem_927, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_168 = torch.ops.aten.where.self(le_scalar_103, new_zeros_default_168, getitem_927);  new_zeros_default_168 = getitem_927 = None
        add_tensor_172 = torch.ops.aten.add.Tensor(where_self_156, where_self_168);  where_self_156 = where_self_168 = None
        slice_tensor_84 = torch.ops.aten.slice.Tensor(where_self_167, 1, 0, 168)
        slice_tensor_85 = torch.ops.aten.slice.Tensor(where_self_167, 1, 168, 336)
        slice_tensor_86 = torch.ops.aten.slice.Tensor(where_self_167, 1, 336, 504)
        slice_tensor_87 = torch.ops.aten.slice.Tensor(where_self_167, 1, 504, 672)
        slice_tensor_88 = torch.ops.aten.slice.Tensor(where_self_167, 1, 672, 840)
        slice_tensor_89 = torch.ops.aten.slice.Tensor(where_self_167, 1, 840, 1008);  where_self_167 = None
        sum_dim_int_list_310 = torch.ops.aten.sum.dim_IntList(slice_tensor_89, [0, 2, 3])
        sub_tensor_540 = torch.ops.aten.sub.Tensor(arg1139_1, arg1896_1);  arg1139_1 = arg1896_1 = None
        mul_tensor_1558 = torch.ops.aten.mul.Tensor(slice_tensor_89, sub_tensor_540)
        sum_dim_int_list_311 = torch.ops.aten.sum.dim_IntList(mul_tensor_1558, [0, 2, 3]);  mul_tensor_1558 = None
        mul_tensor_1559 = torch.ops.aten.mul.Tensor(sum_dim_int_list_310, 0.0005668934240362812);  sum_dim_int_list_310 = None
        view_default_503 = torch.ops.aten.view.default(mul_tensor_1559, [1, 168, 1, 1]);  mul_tensor_1559 = None
        mul_tensor_1560 = torch.ops.aten.mul.Tensor(sum_dim_int_list_311, 0.0005668934240362812)
        mul_tensor_1561 = torch.ops.aten.mul.Tensor(arg1140_1, arg1140_1)
        mul_tensor_1562 = torch.ops.aten.mul.Tensor(mul_tensor_1560, mul_tensor_1561);  mul_tensor_1560 = mul_tensor_1561 = None
        view_default_504 = torch.ops.aten.view.default(mul_tensor_1562, [1, 168, 1, 1]);  mul_tensor_1562 = None
        mul_tensor_1563 = torch.ops.aten.mul.Tensor(arg1140_1, arg292_1);  arg292_1 = None
        view_default_505 = torch.ops.aten.view.default(mul_tensor_1563, [1, 168, 1, 1]);  mul_tensor_1563 = None
        mul_tensor_1564 = torch.ops.aten.mul.Tensor(sub_tensor_540, view_default_504);  sub_tensor_540 = view_default_504 = None
        sub_tensor_541 = torch.ops.aten.sub.Tensor(slice_tensor_89, mul_tensor_1564);  mul_tensor_1564 = None
        sub_tensor_542 = torch.ops.aten.sub.Tensor(sub_tensor_541, view_default_503);  sub_tensor_541 = view_default_503 = None
        mul_tensor_1565 = torch.ops.aten.mul.Tensor(sub_tensor_542, view_default_505);  sub_tensor_542 = view_default_505 = None
        mul_tensor_1566 = torch.ops.aten.mul.Tensor(sum_dim_int_list_311, arg1140_1);  sum_dim_int_list_311 = arg1140_1 = None
        convolution_backward_default_310 = torch.ops.aten.convolution_backward.default(mul_tensor_1565, arg1138_1, arg291_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1565 = arg1138_1 = arg291_1 = None
        getitem_930 = convolution_backward_default_310[0]
        getitem_931 = convolution_backward_default_310[1];  convolution_backward_default_310 = None
        convolution_backward_default_311 = torch.ops.aten.convolution_backward.default(getitem_930, arg1137_1, arg290_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_930 = arg290_1 = None
        getitem_933 = convolution_backward_default_311[0]
        getitem_934 = convolution_backward_default_311[1];  convolution_backward_default_311 = None
        le_scalar_111 = torch.ops.aten.le.Scalar(arg1137_1, 0);  arg1137_1 = None
        new_zeros_default_169 = torch.ops.aten.new_zeros.default(getitem_933, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_169 = torch.ops.aten.where.self(le_scalar_111, new_zeros_default_169, getitem_933);  le_scalar_111 = new_zeros_default_169 = getitem_933 = None
        sum_dim_int_list_312 = torch.ops.aten.sum.dim_IntList(where_self_169, [0, 2, 3])
        sub_tensor_543 = torch.ops.aten.sub.Tensor(arg1135_1, arg1897_1);  arg1135_1 = arg1897_1 = None
        mul_tensor_1567 = torch.ops.aten.mul.Tensor(where_self_169, sub_tensor_543)
        sum_dim_int_list_313 = torch.ops.aten.sum.dim_IntList(mul_tensor_1567, [0, 2, 3]);  mul_tensor_1567 = None
        mul_tensor_1568 = torch.ops.aten.mul.Tensor(sum_dim_int_list_312, 0.0005668934240362812);  sum_dim_int_list_312 = None
        view_default_506 = torch.ops.aten.view.default(mul_tensor_1568, [1, 168, 1, 1]);  mul_tensor_1568 = None
        mul_tensor_1569 = torch.ops.aten.mul.Tensor(sum_dim_int_list_313, 0.0005668934240362812)
        mul_tensor_1570 = torch.ops.aten.mul.Tensor(arg1136_1, arg1136_1)
        mul_tensor_1571 = torch.ops.aten.mul.Tensor(mul_tensor_1569, mul_tensor_1570);  mul_tensor_1569 = mul_tensor_1570 = None
        view_default_507 = torch.ops.aten.view.default(mul_tensor_1571, [1, 168, 1, 1]);  mul_tensor_1571 = None
        mul_tensor_1572 = torch.ops.aten.mul.Tensor(arg1136_1, arg289_1);  arg289_1 = None
        view_default_508 = torch.ops.aten.view.default(mul_tensor_1572, [1, 168, 1, 1]);  mul_tensor_1572 = None
        mul_tensor_1573 = torch.ops.aten.mul.Tensor(sub_tensor_543, view_default_507);  sub_tensor_543 = view_default_507 = None
        sub_tensor_544 = torch.ops.aten.sub.Tensor(where_self_169, mul_tensor_1573);  where_self_169 = mul_tensor_1573 = None
        sub_tensor_545 = torch.ops.aten.sub.Tensor(sub_tensor_544, view_default_506);  sub_tensor_544 = view_default_506 = None
        mul_tensor_1574 = torch.ops.aten.mul.Tensor(sub_tensor_545, view_default_508);  sub_tensor_545 = view_default_508 = None
        mul_tensor_1575 = torch.ops.aten.mul.Tensor(sum_dim_int_list_313, arg1136_1);  sum_dim_int_list_313 = arg1136_1 = None
        convolution_backward_default_312 = torch.ops.aten.convolution_backward.default(mul_tensor_1574, arg1134_1, arg288_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1574 = arg1134_1 = arg288_1 = None
        getitem_936 = convolution_backward_default_312[0]
        getitem_937 = convolution_backward_default_312[1];  convolution_backward_default_312 = None
        convolution_backward_default_313 = torch.ops.aten.convolution_backward.default(getitem_936, relu_default_12, arg287_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_936 = arg287_1 = None
        getitem_939 = convolution_backward_default_313[0]
        getitem_940 = convolution_backward_default_313[1];  convolution_backward_default_313 = None
        le_scalar_112 = torch.ops.aten.le.Scalar(relu_default_12, 0)
        new_zeros_default_170 = torch.ops.aten.new_zeros.default(getitem_939, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_170 = torch.ops.aten.where.self(le_scalar_112, new_zeros_default_170, getitem_939);  new_zeros_default_170 = getitem_939 = None
        add_tensor_173 = torch.ops.aten.add.Tensor(slice_tensor_89, where_self_170);  slice_tensor_89 = where_self_170 = None
        avg_pool2d_backward_default_32 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_88, add_tensor_10, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_88 = add_tensor_10 = None
        add_tensor_174 = torch.ops.aten.add.Tensor(slice_tensor_84, avg_pool2d_backward_default_32);  slice_tensor_84 = None
        add_tensor_175 = torch.ops.aten.add.Tensor(add_tensor_174, avg_pool2d_backward_default_32);  add_tensor_174 = avg_pool2d_backward_default_32 = None
        add_tensor_176 = torch.ops.aten.add.Tensor(add_tensor_175, slice_tensor_87);  add_tensor_175 = None
        avg_pool2d_backward_default_33 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_87, add_tensor_11, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_87 = add_tensor_11 = None
        add_tensor_177 = torch.ops.aten.add.Tensor(add_tensor_173, avg_pool2d_backward_default_33);  add_tensor_173 = avg_pool2d_backward_default_33 = None
        sum_dim_int_list_314 = torch.ops.aten.sum.dim_IntList(slice_tensor_86, [0, 2, 3])
        sub_tensor_546 = torch.ops.aten.sub.Tensor(arg1132_1, arg1898_1);  arg1132_1 = arg1898_1 = None
        mul_tensor_1576 = torch.ops.aten.mul.Tensor(slice_tensor_86, sub_tensor_546)
        sum_dim_int_list_315 = torch.ops.aten.sum.dim_IntList(mul_tensor_1576, [0, 2, 3]);  mul_tensor_1576 = None
        mul_tensor_1577 = torch.ops.aten.mul.Tensor(sum_dim_int_list_314, 0.0005668934240362812);  sum_dim_int_list_314 = None
        view_default_509 = torch.ops.aten.view.default(mul_tensor_1577, [1, 168, 1, 1]);  mul_tensor_1577 = None
        mul_tensor_1578 = torch.ops.aten.mul.Tensor(sum_dim_int_list_315, 0.0005668934240362812)
        mul_tensor_1579 = torch.ops.aten.mul.Tensor(arg1133_1, arg1133_1)
        mul_tensor_1580 = torch.ops.aten.mul.Tensor(mul_tensor_1578, mul_tensor_1579);  mul_tensor_1578 = mul_tensor_1579 = None
        view_default_510 = torch.ops.aten.view.default(mul_tensor_1580, [1, 168, 1, 1]);  mul_tensor_1580 = None
        mul_tensor_1581 = torch.ops.aten.mul.Tensor(arg1133_1, arg286_1);  arg286_1 = None
        view_default_511 = torch.ops.aten.view.default(mul_tensor_1581, [1, 168, 1, 1]);  mul_tensor_1581 = None
        mul_tensor_1582 = torch.ops.aten.mul.Tensor(sub_tensor_546, view_default_510);  sub_tensor_546 = view_default_510 = None
        sub_tensor_547 = torch.ops.aten.sub.Tensor(slice_tensor_86, mul_tensor_1582);  mul_tensor_1582 = None
        sub_tensor_548 = torch.ops.aten.sub.Tensor(sub_tensor_547, view_default_509);  sub_tensor_547 = None
        mul_tensor_1583 = torch.ops.aten.mul.Tensor(sub_tensor_548, view_default_511);  sub_tensor_548 = view_default_511 = None
        mul_tensor_1584 = torch.ops.aten.mul.Tensor(sum_dim_int_list_315, arg1133_1);  sum_dim_int_list_315 = arg1133_1 = None
        convolution_backward_default_314 = torch.ops.aten.convolution_backward.default(mul_tensor_1583, arg1131_1, arg285_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1583 = arg1131_1 = arg285_1 = None
        getitem_942 = convolution_backward_default_314[0]
        getitem_943 = convolution_backward_default_314[1];  convolution_backward_default_314 = None
        convolution_backward_default_315 = torch.ops.aten.convolution_backward.default(getitem_942, arg1130_1, arg284_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_942 = arg284_1 = None
        getitem_945 = convolution_backward_default_315[0]
        getitem_946 = convolution_backward_default_315[1];  convolution_backward_default_315 = None
        le_scalar_113 = torch.ops.aten.le.Scalar(arg1130_1, 0);  arg1130_1 = None
        new_zeros_default_171 = torch.ops.aten.new_zeros.default(getitem_945, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_171 = torch.ops.aten.where.self(le_scalar_113, new_zeros_default_171, getitem_945);  le_scalar_113 = new_zeros_default_171 = getitem_945 = None
        sum_dim_int_list_316 = torch.ops.aten.sum.dim_IntList(where_self_171, [0, 2, 3])
        sub_tensor_549 = torch.ops.aten.sub.Tensor(arg1128_1, arg1899_1);  arg1128_1 = arg1899_1 = None
        mul_tensor_1585 = torch.ops.aten.mul.Tensor(where_self_171, sub_tensor_549)
        sum_dim_int_list_317 = torch.ops.aten.sum.dim_IntList(mul_tensor_1585, [0, 2, 3]);  mul_tensor_1585 = None
        mul_tensor_1586 = torch.ops.aten.mul.Tensor(sum_dim_int_list_316, 0.0005668934240362812);  sum_dim_int_list_316 = None
        view_default_512 = torch.ops.aten.view.default(mul_tensor_1586, [1, 168, 1, 1]);  mul_tensor_1586 = None
        mul_tensor_1587 = torch.ops.aten.mul.Tensor(sum_dim_int_list_317, 0.0005668934240362812)
        mul_tensor_1588 = torch.ops.aten.mul.Tensor(arg1129_1, arg1129_1)
        mul_tensor_1589 = torch.ops.aten.mul.Tensor(mul_tensor_1587, mul_tensor_1588);  mul_tensor_1587 = mul_tensor_1588 = None
        view_default_513 = torch.ops.aten.view.default(mul_tensor_1589, [1, 168, 1, 1]);  mul_tensor_1589 = None
        mul_tensor_1590 = torch.ops.aten.mul.Tensor(arg1129_1, arg283_1);  arg283_1 = None
        view_default_514 = torch.ops.aten.view.default(mul_tensor_1590, [1, 168, 1, 1]);  mul_tensor_1590 = None
        mul_tensor_1591 = torch.ops.aten.mul.Tensor(sub_tensor_549, view_default_513);  sub_tensor_549 = view_default_513 = None
        sub_tensor_550 = torch.ops.aten.sub.Tensor(where_self_171, mul_tensor_1591);  where_self_171 = mul_tensor_1591 = None
        sub_tensor_551 = torch.ops.aten.sub.Tensor(sub_tensor_550, view_default_512);  sub_tensor_550 = view_default_512 = None
        mul_tensor_1592 = torch.ops.aten.mul.Tensor(sub_tensor_551, view_default_514);  sub_tensor_551 = view_default_514 = None
        mul_tensor_1593 = torch.ops.aten.mul.Tensor(sum_dim_int_list_317, arg1129_1);  sum_dim_int_list_317 = arg1129_1 = None
        convolution_backward_default_316 = torch.ops.aten.convolution_backward.default(mul_tensor_1592, arg1127_1, arg282_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1592 = arg1127_1 = arg282_1 = None
        getitem_948 = convolution_backward_default_316[0]
        getitem_949 = convolution_backward_default_316[1];  convolution_backward_default_316 = None
        convolution_backward_default_317 = torch.ops.aten.convolution_backward.default(getitem_948, relu_default_13, arg281_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_948 = arg281_1 = None
        getitem_951 = convolution_backward_default_317[0]
        getitem_952 = convolution_backward_default_317[1];  convolution_backward_default_317 = None
        le_scalar_114 = torch.ops.aten.le.Scalar(relu_default_13, 0)
        new_zeros_default_172 = torch.ops.aten.new_zeros.default(getitem_951, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_172 = torch.ops.aten.where.self(le_scalar_114, new_zeros_default_172, getitem_951);  new_zeros_default_172 = getitem_951 = None
        add_tensor_178 = torch.ops.aten.add.Tensor(add_tensor_176, where_self_172);  add_tensor_176 = where_self_172 = None
        sub_tensor_552 = torch.ops.aten.sub.Tensor(arg1125_1, arg1900_1);  arg1125_1 = arg1900_1 = None
        mul_tensor_1594 = torch.ops.aten.mul.Tensor(slice_tensor_86, sub_tensor_552)
        sum_dim_int_list_318 = torch.ops.aten.sum.dim_IntList(mul_tensor_1594, [0, 2, 3]);  mul_tensor_1594 = None
        mul_tensor_1595 = torch.ops.aten.mul.Tensor(sum_dim_int_list_318, 0.0005668934240362812)
        mul_tensor_1596 = torch.ops.aten.mul.Tensor(arg1126_1, arg1126_1)
        mul_tensor_1597 = torch.ops.aten.mul.Tensor(mul_tensor_1595, mul_tensor_1596);  mul_tensor_1595 = mul_tensor_1596 = None
        view_default_515 = torch.ops.aten.view.default(mul_tensor_1597, [1, 168, 1, 1]);  mul_tensor_1597 = None
        mul_tensor_1598 = torch.ops.aten.mul.Tensor(arg1126_1, arg280_1);  arg280_1 = None
        view_default_516 = torch.ops.aten.view.default(mul_tensor_1598, [1, 168, 1, 1]);  mul_tensor_1598 = None
        mul_tensor_1599 = torch.ops.aten.mul.Tensor(sub_tensor_552, view_default_515);  sub_tensor_552 = view_default_515 = None
        sub_tensor_553 = torch.ops.aten.sub.Tensor(slice_tensor_86, mul_tensor_1599);  slice_tensor_86 = mul_tensor_1599 = None
        sub_tensor_554 = torch.ops.aten.sub.Tensor(sub_tensor_553, view_default_509);  sub_tensor_553 = view_default_509 = None
        mul_tensor_1600 = torch.ops.aten.mul.Tensor(sub_tensor_554, view_default_516);  sub_tensor_554 = view_default_516 = None
        mul_tensor_1601 = torch.ops.aten.mul.Tensor(sum_dim_int_list_318, arg1126_1);  sum_dim_int_list_318 = arg1126_1 = None
        convolution_backward_default_318 = torch.ops.aten.convolution_backward.default(mul_tensor_1600, arg1124_1, arg279_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1600 = arg1124_1 = arg279_1 = None
        getitem_954 = convolution_backward_default_318[0]
        getitem_955 = convolution_backward_default_318[1];  convolution_backward_default_318 = None
        convolution_backward_default_319 = torch.ops.aten.convolution_backward.default(getitem_954, arg1123_1, arg278_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_954 = arg278_1 = None
        getitem_957 = convolution_backward_default_319[0]
        getitem_958 = convolution_backward_default_319[1];  convolution_backward_default_319 = None
        le_scalar_115 = torch.ops.aten.le.Scalar(arg1123_1, 0);  arg1123_1 = None
        new_zeros_default_173 = torch.ops.aten.new_zeros.default(getitem_957, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_173 = torch.ops.aten.where.self(le_scalar_115, new_zeros_default_173, getitem_957);  le_scalar_115 = new_zeros_default_173 = getitem_957 = None
        sum_dim_int_list_319 = torch.ops.aten.sum.dim_IntList(where_self_173, [0, 2, 3])
        sub_tensor_555 = torch.ops.aten.sub.Tensor(arg1121_1, arg1901_1);  arg1121_1 = arg1901_1 = None
        mul_tensor_1602 = torch.ops.aten.mul.Tensor(where_self_173, sub_tensor_555)
        sum_dim_int_list_320 = torch.ops.aten.sum.dim_IntList(mul_tensor_1602, [0, 2, 3]);  mul_tensor_1602 = None
        mul_tensor_1603 = torch.ops.aten.mul.Tensor(sum_dim_int_list_319, 0.0005668934240362812);  sum_dim_int_list_319 = None
        view_default_517 = torch.ops.aten.view.default(mul_tensor_1603, [1, 168, 1, 1]);  mul_tensor_1603 = None
        mul_tensor_1604 = torch.ops.aten.mul.Tensor(sum_dim_int_list_320, 0.0005668934240362812)
        mul_tensor_1605 = torch.ops.aten.mul.Tensor(arg1122_1, arg1122_1)
        mul_tensor_1606 = torch.ops.aten.mul.Tensor(mul_tensor_1604, mul_tensor_1605);  mul_tensor_1604 = mul_tensor_1605 = None
        view_default_518 = torch.ops.aten.view.default(mul_tensor_1606, [1, 168, 1, 1]);  mul_tensor_1606 = None
        mul_tensor_1607 = torch.ops.aten.mul.Tensor(arg1122_1, arg277_1);  arg277_1 = None
        view_default_519 = torch.ops.aten.view.default(mul_tensor_1607, [1, 168, 1, 1]);  mul_tensor_1607 = None
        mul_tensor_1608 = torch.ops.aten.mul.Tensor(sub_tensor_555, view_default_518);  sub_tensor_555 = view_default_518 = None
        sub_tensor_556 = torch.ops.aten.sub.Tensor(where_self_173, mul_tensor_1608);  where_self_173 = mul_tensor_1608 = None
        sub_tensor_557 = torch.ops.aten.sub.Tensor(sub_tensor_556, view_default_517);  sub_tensor_556 = view_default_517 = None
        mul_tensor_1609 = torch.ops.aten.mul.Tensor(sub_tensor_557, view_default_519);  sub_tensor_557 = view_default_519 = None
        mul_tensor_1610 = torch.ops.aten.mul.Tensor(sum_dim_int_list_320, arg1122_1);  sum_dim_int_list_320 = arg1122_1 = None
        convolution_backward_default_320 = torch.ops.aten.convolution_backward.default(mul_tensor_1609, arg1120_1, arg276_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1609 = arg1120_1 = arg276_1 = None
        getitem_960 = convolution_backward_default_320[0]
        getitem_961 = convolution_backward_default_320[1];  convolution_backward_default_320 = None
        convolution_backward_default_321 = torch.ops.aten.convolution_backward.default(getitem_960, relu_default_13, arg275_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_960 = arg275_1 = None
        getitem_963 = convolution_backward_default_321[0]
        getitem_964 = convolution_backward_default_321[1];  convolution_backward_default_321 = None
        new_zeros_default_174 = torch.ops.aten.new_zeros.default(getitem_963, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_174 = torch.ops.aten.where.self(le_scalar_114, new_zeros_default_174, getitem_963);  new_zeros_default_174 = getitem_963 = None
        add_tensor_179 = torch.ops.aten.add.Tensor(add_tensor_178, where_self_174);  add_tensor_178 = where_self_174 = None
        sum_dim_int_list_321 = torch.ops.aten.sum.dim_IntList(slice_tensor_85, [0, 2, 3])
        sub_tensor_558 = torch.ops.aten.sub.Tensor(arg1118_1, arg1902_1);  arg1118_1 = arg1902_1 = None
        mul_tensor_1611 = torch.ops.aten.mul.Tensor(slice_tensor_85, sub_tensor_558)
        sum_dim_int_list_322 = torch.ops.aten.sum.dim_IntList(mul_tensor_1611, [0, 2, 3]);  mul_tensor_1611 = None
        mul_tensor_1612 = torch.ops.aten.mul.Tensor(sum_dim_int_list_321, 0.0005668934240362812);  sum_dim_int_list_321 = None
        view_default_520 = torch.ops.aten.view.default(mul_tensor_1612, [1, 168, 1, 1]);  mul_tensor_1612 = None
        mul_tensor_1613 = torch.ops.aten.mul.Tensor(sum_dim_int_list_322, 0.0005668934240362812)
        mul_tensor_1614 = torch.ops.aten.mul.Tensor(arg1119_1, arg1119_1)
        mul_tensor_1615 = torch.ops.aten.mul.Tensor(mul_tensor_1613, mul_tensor_1614);  mul_tensor_1613 = mul_tensor_1614 = None
        view_default_521 = torch.ops.aten.view.default(mul_tensor_1615, [1, 168, 1, 1]);  mul_tensor_1615 = None
        mul_tensor_1616 = torch.ops.aten.mul.Tensor(arg1119_1, arg274_1);  arg274_1 = None
        view_default_522 = torch.ops.aten.view.default(mul_tensor_1616, [1, 168, 1, 1]);  mul_tensor_1616 = None
        mul_tensor_1617 = torch.ops.aten.mul.Tensor(sub_tensor_558, view_default_521);  sub_tensor_558 = view_default_521 = None
        sub_tensor_559 = torch.ops.aten.sub.Tensor(slice_tensor_85, mul_tensor_1617);  mul_tensor_1617 = None
        sub_tensor_560 = torch.ops.aten.sub.Tensor(sub_tensor_559, view_default_520);  sub_tensor_559 = None
        mul_tensor_1618 = torch.ops.aten.mul.Tensor(sub_tensor_560, view_default_522);  sub_tensor_560 = view_default_522 = None
        mul_tensor_1619 = torch.ops.aten.mul.Tensor(sum_dim_int_list_322, arg1119_1);  sum_dim_int_list_322 = arg1119_1 = None
        convolution_backward_default_322 = torch.ops.aten.convolution_backward.default(mul_tensor_1618, arg1117_1, arg273_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1618 = arg1117_1 = arg273_1 = None
        getitem_966 = convolution_backward_default_322[0]
        getitem_967 = convolution_backward_default_322[1];  convolution_backward_default_322 = None
        convolution_backward_default_323 = torch.ops.aten.convolution_backward.default(getitem_966, arg1116_1, arg272_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_966 = arg272_1 = None
        getitem_969 = convolution_backward_default_323[0]
        getitem_970 = convolution_backward_default_323[1];  convolution_backward_default_323 = None
        le_scalar_116 = torch.ops.aten.le.Scalar(arg1116_1, 0);  arg1116_1 = None
        new_zeros_default_175 = torch.ops.aten.new_zeros.default(getitem_969, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_175 = torch.ops.aten.where.self(le_scalar_116, new_zeros_default_175, getitem_969);  le_scalar_116 = new_zeros_default_175 = getitem_969 = None
        sum_dim_int_list_323 = torch.ops.aten.sum.dim_IntList(where_self_175, [0, 2, 3])
        sub_tensor_561 = torch.ops.aten.sub.Tensor(arg1114_1, arg1903_1);  arg1114_1 = arg1903_1 = None
        mul_tensor_1620 = torch.ops.aten.mul.Tensor(where_self_175, sub_tensor_561)
        sum_dim_int_list_324 = torch.ops.aten.sum.dim_IntList(mul_tensor_1620, [0, 2, 3]);  mul_tensor_1620 = None
        mul_tensor_1621 = torch.ops.aten.mul.Tensor(sum_dim_int_list_323, 0.0005668934240362812);  sum_dim_int_list_323 = None
        view_default_523 = torch.ops.aten.view.default(mul_tensor_1621, [1, 168, 1, 1]);  mul_tensor_1621 = None
        mul_tensor_1622 = torch.ops.aten.mul.Tensor(sum_dim_int_list_324, 0.0005668934240362812)
        mul_tensor_1623 = torch.ops.aten.mul.Tensor(arg1115_1, arg1115_1)
        mul_tensor_1624 = torch.ops.aten.mul.Tensor(mul_tensor_1622, mul_tensor_1623);  mul_tensor_1622 = mul_tensor_1623 = None
        view_default_524 = torch.ops.aten.view.default(mul_tensor_1624, [1, 168, 1, 1]);  mul_tensor_1624 = None
        mul_tensor_1625 = torch.ops.aten.mul.Tensor(arg1115_1, arg271_1);  arg271_1 = None
        view_default_525 = torch.ops.aten.view.default(mul_tensor_1625, [1, 168, 1, 1]);  mul_tensor_1625 = None
        mul_tensor_1626 = torch.ops.aten.mul.Tensor(sub_tensor_561, view_default_524);  sub_tensor_561 = view_default_524 = None
        sub_tensor_562 = torch.ops.aten.sub.Tensor(where_self_175, mul_tensor_1626);  where_self_175 = mul_tensor_1626 = None
        sub_tensor_563 = torch.ops.aten.sub.Tensor(sub_tensor_562, view_default_523);  sub_tensor_562 = view_default_523 = None
        mul_tensor_1627 = torch.ops.aten.mul.Tensor(sub_tensor_563, view_default_525);  sub_tensor_563 = view_default_525 = None
        mul_tensor_1628 = torch.ops.aten.mul.Tensor(sum_dim_int_list_324, arg1115_1);  sum_dim_int_list_324 = arg1115_1 = None
        convolution_backward_default_324 = torch.ops.aten.convolution_backward.default(mul_tensor_1627, arg1113_1, arg270_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1627 = arg1113_1 = arg270_1 = None
        getitem_972 = convolution_backward_default_324[0]
        getitem_973 = convolution_backward_default_324[1];  convolution_backward_default_324 = None
        convolution_backward_default_325 = torch.ops.aten.convolution_backward.default(getitem_972, relu_default_13, arg269_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_972 = relu_default_13 = arg269_1 = None
        getitem_975 = convolution_backward_default_325[0]
        getitem_976 = convolution_backward_default_325[1];  convolution_backward_default_325 = None
        new_zeros_default_176 = torch.ops.aten.new_zeros.default(getitem_975, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_176 = torch.ops.aten.where.self(le_scalar_114, new_zeros_default_176, getitem_975);  le_scalar_114 = new_zeros_default_176 = getitem_975 = None
        add_tensor_180 = torch.ops.aten.add.Tensor(add_tensor_179, where_self_176);  add_tensor_179 = where_self_176 = None
        sub_tensor_564 = torch.ops.aten.sub.Tensor(arg1111_1, arg1904_1);  arg1111_1 = arg1904_1 = None
        mul_tensor_1629 = torch.ops.aten.mul.Tensor(slice_tensor_85, sub_tensor_564)
        sum_dim_int_list_325 = torch.ops.aten.sum.dim_IntList(mul_tensor_1629, [0, 2, 3]);  mul_tensor_1629 = None
        mul_tensor_1630 = torch.ops.aten.mul.Tensor(sum_dim_int_list_325, 0.0005668934240362812)
        mul_tensor_1631 = torch.ops.aten.mul.Tensor(arg1112_1, arg1112_1)
        mul_tensor_1632 = torch.ops.aten.mul.Tensor(mul_tensor_1630, mul_tensor_1631);  mul_tensor_1630 = mul_tensor_1631 = None
        view_default_526 = torch.ops.aten.view.default(mul_tensor_1632, [1, 168, 1, 1]);  mul_tensor_1632 = None
        mul_tensor_1633 = torch.ops.aten.mul.Tensor(arg1112_1, arg268_1);  arg268_1 = None
        view_default_527 = torch.ops.aten.view.default(mul_tensor_1633, [1, 168, 1, 1]);  mul_tensor_1633 = None
        mul_tensor_1634 = torch.ops.aten.mul.Tensor(sub_tensor_564, view_default_526);  sub_tensor_564 = view_default_526 = None
        sub_tensor_565 = torch.ops.aten.sub.Tensor(slice_tensor_85, mul_tensor_1634);  slice_tensor_85 = mul_tensor_1634 = None
        sub_tensor_566 = torch.ops.aten.sub.Tensor(sub_tensor_565, view_default_520);  sub_tensor_565 = view_default_520 = None
        mul_tensor_1635 = torch.ops.aten.mul.Tensor(sub_tensor_566, view_default_527);  sub_tensor_566 = view_default_527 = None
        mul_tensor_1636 = torch.ops.aten.mul.Tensor(sum_dim_int_list_325, arg1112_1);  sum_dim_int_list_325 = arg1112_1 = None
        convolution_backward_default_326 = torch.ops.aten.convolution_backward.default(mul_tensor_1635, arg1110_1, arg267_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1635 = arg1110_1 = arg267_1 = None
        getitem_978 = convolution_backward_default_326[0]
        getitem_979 = convolution_backward_default_326[1];  convolution_backward_default_326 = None
        convolution_backward_default_327 = torch.ops.aten.convolution_backward.default(getitem_978, arg1109_1, arg266_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_978 = arg266_1 = None
        getitem_981 = convolution_backward_default_327[0]
        getitem_982 = convolution_backward_default_327[1];  convolution_backward_default_327 = None
        le_scalar_117 = torch.ops.aten.le.Scalar(arg1109_1, 0);  arg1109_1 = None
        new_zeros_default_177 = torch.ops.aten.new_zeros.default(getitem_981, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_177 = torch.ops.aten.where.self(le_scalar_117, new_zeros_default_177, getitem_981);  le_scalar_117 = new_zeros_default_177 = getitem_981 = None
        sum_dim_int_list_326 = torch.ops.aten.sum.dim_IntList(where_self_177, [0, 2, 3])
        sub_tensor_567 = torch.ops.aten.sub.Tensor(arg1107_1, arg1905_1);  arg1107_1 = arg1905_1 = None
        mul_tensor_1637 = torch.ops.aten.mul.Tensor(where_self_177, sub_tensor_567)
        sum_dim_int_list_327 = torch.ops.aten.sum.dim_IntList(mul_tensor_1637, [0, 2, 3]);  mul_tensor_1637 = None
        mul_tensor_1638 = torch.ops.aten.mul.Tensor(sum_dim_int_list_326, 0.0005668934240362812);  sum_dim_int_list_326 = None
        view_default_528 = torch.ops.aten.view.default(mul_tensor_1638, [1, 168, 1, 1]);  mul_tensor_1638 = None
        mul_tensor_1639 = torch.ops.aten.mul.Tensor(sum_dim_int_list_327, 0.0005668934240362812)
        mul_tensor_1640 = torch.ops.aten.mul.Tensor(arg1108_1, arg1108_1)
        mul_tensor_1641 = torch.ops.aten.mul.Tensor(mul_tensor_1639, mul_tensor_1640);  mul_tensor_1639 = mul_tensor_1640 = None
        view_default_529 = torch.ops.aten.view.default(mul_tensor_1641, [1, 168, 1, 1]);  mul_tensor_1641 = None
        mul_tensor_1642 = torch.ops.aten.mul.Tensor(arg1108_1, arg265_1);  arg265_1 = None
        view_default_530 = torch.ops.aten.view.default(mul_tensor_1642, [1, 168, 1, 1]);  mul_tensor_1642 = None
        mul_tensor_1643 = torch.ops.aten.mul.Tensor(sub_tensor_567, view_default_529);  sub_tensor_567 = view_default_529 = None
        sub_tensor_568 = torch.ops.aten.sub.Tensor(where_self_177, mul_tensor_1643);  where_self_177 = mul_tensor_1643 = None
        sub_tensor_569 = torch.ops.aten.sub.Tensor(sub_tensor_568, view_default_528);  sub_tensor_568 = view_default_528 = None
        mul_tensor_1644 = torch.ops.aten.mul.Tensor(sub_tensor_569, view_default_530);  sub_tensor_569 = view_default_530 = None
        mul_tensor_1645 = torch.ops.aten.mul.Tensor(sum_dim_int_list_327, arg1108_1);  sum_dim_int_list_327 = arg1108_1 = None
        convolution_backward_default_328 = torch.ops.aten.convolution_backward.default(mul_tensor_1644, arg1106_1, arg264_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1644 = arg1106_1 = arg264_1 = None
        getitem_984 = convolution_backward_default_328[0]
        getitem_985 = convolution_backward_default_328[1];  convolution_backward_default_328 = None
        convolution_backward_default_329 = torch.ops.aten.convolution_backward.default(getitem_984, relu_default_12, arg263_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_984 = relu_default_12 = arg263_1 = None
        getitem_987 = convolution_backward_default_329[0]
        getitem_988 = convolution_backward_default_329[1];  convolution_backward_default_329 = None
        new_zeros_default_178 = torch.ops.aten.new_zeros.default(getitem_987, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_178 = torch.ops.aten.where.self(le_scalar_112, new_zeros_default_178, getitem_987);  le_scalar_112 = new_zeros_default_178 = getitem_987 = None
        add_tensor_181 = torch.ops.aten.add.Tensor(add_tensor_177, where_self_178);  add_tensor_177 = where_self_178 = None
        view_default_531 = torch.ops.aten.view.default(squeeze_dim_68, [1, 168, 1, 1]);  squeeze_dim_68 = None
        sum_dim_int_list_328 = torch.ops.aten.sum.dim_IntList(add_tensor_181, [0, 2, 3])
        sub_tensor_570 = torch.ops.aten.sub.Tensor(arg1103_1, view_default_531);  arg1103_1 = view_default_531 = None
        mul_tensor_1646 = torch.ops.aten.mul.Tensor(add_tensor_181, sub_tensor_570)
        sum_dim_int_list_329 = torch.ops.aten.sum.dim_IntList(mul_tensor_1646, [0, 2, 3]);  mul_tensor_1646 = None
        mul_tensor_1647 = torch.ops.aten.mul.Tensor(sum_dim_int_list_328, 0.0005668934240362812);  sum_dim_int_list_328 = None
        view_default_532 = torch.ops.aten.view.default(mul_tensor_1647, [1, 168, 1, 1]);  mul_tensor_1647 = None
        mul_tensor_1648 = torch.ops.aten.mul.Tensor(sum_dim_int_list_329, 0.0005668934240362812)
        mul_tensor_1649 = torch.ops.aten.mul.Tensor(squeeze_dim_71, squeeze_dim_71)
        mul_tensor_1650 = torch.ops.aten.mul.Tensor(mul_tensor_1648, mul_tensor_1649);  mul_tensor_1648 = mul_tensor_1649 = None
        view_default_533 = torch.ops.aten.view.default(mul_tensor_1650, [1, 168, 1, 1]);  mul_tensor_1650 = None
        mul_tensor_1651 = torch.ops.aten.mul.Tensor(squeeze_dim_71, arg261_1);  arg261_1 = None
        view_default_534 = torch.ops.aten.view.default(mul_tensor_1651, [1, 168, 1, 1]);  mul_tensor_1651 = None
        mul_tensor_1652 = torch.ops.aten.mul.Tensor(sub_tensor_570, view_default_533);  sub_tensor_570 = view_default_533 = None
        sub_tensor_571 = torch.ops.aten.sub.Tensor(add_tensor_181, mul_tensor_1652);  add_tensor_181 = mul_tensor_1652 = None
        sub_tensor_572 = torch.ops.aten.sub.Tensor(sub_tensor_571, view_default_532);  sub_tensor_571 = view_default_532 = None
        mul_tensor_1653 = torch.ops.aten.mul.Tensor(sub_tensor_572, view_default_534);  sub_tensor_572 = view_default_534 = None
        mul_tensor_1654 = torch.ops.aten.mul.Tensor(sum_dim_int_list_329, squeeze_dim_71);  sum_dim_int_list_329 = squeeze_dim_71 = None
        convolution_backward_default_330 = torch.ops.aten.convolution_backward.default(mul_tensor_1653, arg1102_1, arg260_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1653 = arg1102_1 = arg260_1 = None
        getitem_990 = convolution_backward_default_330[0]
        getitem_991 = convolution_backward_default_330[1];  convolution_backward_default_330 = None
        new_zeros_default_179 = torch.ops.aten.new_zeros.default(getitem_990, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_179 = torch.ops.aten.where.self(le_scalar_103, new_zeros_default_179, getitem_990);  le_scalar_103 = new_zeros_default_179 = getitem_990 = None
        add_tensor_182 = torch.ops.aten.add.Tensor(add_tensor_172, where_self_179);  add_tensor_172 = where_self_179 = None
        view_default_535 = torch.ops.aten.view.default(squeeze_dim_62, [1, 168, 1, 1]);  squeeze_dim_62 = None
        sum_dim_int_list_330 = torch.ops.aten.sum.dim_IntList(add_tensor_180, [0, 2, 3])
        sub_tensor_573 = torch.ops.aten.sub.Tensor(arg1099_1, view_default_535);  arg1099_1 = view_default_535 = None
        mul_tensor_1655 = torch.ops.aten.mul.Tensor(add_tensor_180, sub_tensor_573)
        sum_dim_int_list_331 = torch.ops.aten.sum.dim_IntList(mul_tensor_1655, [0, 2, 3]);  mul_tensor_1655 = None
        mul_tensor_1656 = torch.ops.aten.mul.Tensor(sum_dim_int_list_330, 0.0005668934240362812);  sum_dim_int_list_330 = None
        view_default_536 = torch.ops.aten.view.default(mul_tensor_1656, [1, 168, 1, 1]);  mul_tensor_1656 = None
        mul_tensor_1657 = torch.ops.aten.mul.Tensor(sum_dim_int_list_331, 0.0005668934240362812)
        mul_tensor_1658 = torch.ops.aten.mul.Tensor(squeeze_dim_65, squeeze_dim_65)
        mul_tensor_1659 = torch.ops.aten.mul.Tensor(mul_tensor_1657, mul_tensor_1658);  mul_tensor_1657 = mul_tensor_1658 = None
        view_default_537 = torch.ops.aten.view.default(mul_tensor_1659, [1, 168, 1, 1]);  mul_tensor_1659 = None
        mul_tensor_1660 = torch.ops.aten.mul.Tensor(squeeze_dim_65, arg258_1);  arg258_1 = None
        view_default_538 = torch.ops.aten.view.default(mul_tensor_1660, [1, 168, 1, 1]);  mul_tensor_1660 = None
        mul_tensor_1661 = torch.ops.aten.mul.Tensor(sub_tensor_573, view_default_537);  sub_tensor_573 = view_default_537 = None
        sub_tensor_574 = torch.ops.aten.sub.Tensor(add_tensor_180, mul_tensor_1661);  add_tensor_180 = mul_tensor_1661 = None
        sub_tensor_575 = torch.ops.aten.sub.Tensor(sub_tensor_574, view_default_536);  sub_tensor_574 = view_default_536 = None
        mul_tensor_1662 = torch.ops.aten.mul.Tensor(sub_tensor_575, view_default_538);  sub_tensor_575 = view_default_538 = None
        mul_tensor_1663 = torch.ops.aten.mul.Tensor(sum_dim_int_list_331, squeeze_dim_65);  sum_dim_int_list_331 = squeeze_dim_65 = None
        convolution_backward_default_331 = torch.ops.aten.convolution_backward.default(mul_tensor_1662, arg1060_1, arg257_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1662 = arg257_1 = None
        getitem_993 = convolution_backward_default_331[0]
        getitem_994 = convolution_backward_default_331[1];  convolution_backward_default_331 = None
        le_scalar_118 = torch.ops.aten.le.Scalar(arg1060_1, 0);  arg1060_1 = None
        new_zeros_default_180 = torch.ops.aten.new_zeros.default(getitem_993, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_180 = torch.ops.aten.where.self(le_scalar_118, new_zeros_default_180, getitem_993);  le_scalar_118 = new_zeros_default_180 = getitem_993 = None
        slice_tensor_90 = torch.ops.aten.slice.Tensor(add_tensor_182, 1, 0, 168)
        slice_tensor_91 = torch.ops.aten.slice.Tensor(add_tensor_182, 1, 168, 336)
        slice_tensor_92 = torch.ops.aten.slice.Tensor(add_tensor_182, 1, 336, 504)
        slice_tensor_93 = torch.ops.aten.slice.Tensor(add_tensor_182, 1, 504, 672)
        slice_tensor_94 = torch.ops.aten.slice.Tensor(add_tensor_182, 1, 672, 840)
        slice_tensor_95 = torch.ops.aten.slice.Tensor(add_tensor_182, 1, 840, 1008);  add_tensor_182 = None
        sum_dim_int_list_332 = torch.ops.aten.sum.dim_IntList(slice_tensor_95, [0, 2, 3])
        sub_tensor_576 = torch.ops.aten.sub.Tensor(arg1097_1, arg1906_1);  arg1097_1 = arg1906_1 = None
        mul_tensor_1664 = torch.ops.aten.mul.Tensor(slice_tensor_95, sub_tensor_576)
        sum_dim_int_list_333 = torch.ops.aten.sum.dim_IntList(mul_tensor_1664, [0, 2, 3]);  mul_tensor_1664 = None
        mul_tensor_1665 = torch.ops.aten.mul.Tensor(sum_dim_int_list_332, 0.0005668934240362812);  sum_dim_int_list_332 = None
        view_default_539 = torch.ops.aten.view.default(mul_tensor_1665, [1, 168, 1, 1]);  mul_tensor_1665 = None
        mul_tensor_1666 = torch.ops.aten.mul.Tensor(sum_dim_int_list_333, 0.0005668934240362812)
        mul_tensor_1667 = torch.ops.aten.mul.Tensor(arg1098_1, arg1098_1)
        mul_tensor_1668 = torch.ops.aten.mul.Tensor(mul_tensor_1666, mul_tensor_1667);  mul_tensor_1666 = mul_tensor_1667 = None
        view_default_540 = torch.ops.aten.view.default(mul_tensor_1668, [1, 168, 1, 1]);  mul_tensor_1668 = None
        mul_tensor_1669 = torch.ops.aten.mul.Tensor(arg1098_1, arg256_1);  arg256_1 = None
        view_default_541 = torch.ops.aten.view.default(mul_tensor_1669, [1, 168, 1, 1]);  mul_tensor_1669 = None
        mul_tensor_1670 = torch.ops.aten.mul.Tensor(sub_tensor_576, view_default_540);  sub_tensor_576 = view_default_540 = None
        sub_tensor_577 = torch.ops.aten.sub.Tensor(slice_tensor_95, mul_tensor_1670);  mul_tensor_1670 = None
        sub_tensor_578 = torch.ops.aten.sub.Tensor(sub_tensor_577, view_default_539);  sub_tensor_577 = view_default_539 = None
        mul_tensor_1671 = torch.ops.aten.mul.Tensor(sub_tensor_578, view_default_541);  sub_tensor_578 = view_default_541 = None
        mul_tensor_1672 = torch.ops.aten.mul.Tensor(sum_dim_int_list_333, arg1098_1);  sum_dim_int_list_333 = arg1098_1 = None
        convolution_backward_default_332 = torch.ops.aten.convolution_backward.default(mul_tensor_1671, arg1096_1, arg255_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1671 = arg1096_1 = arg255_1 = None
        getitem_996 = convolution_backward_default_332[0]
        getitem_997 = convolution_backward_default_332[1];  convolution_backward_default_332 = None
        convolution_backward_default_333 = torch.ops.aten.convolution_backward.default(getitem_996, arg1095_1, arg254_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_996 = arg254_1 = None
        getitem_999 = convolution_backward_default_333[0]
        getitem_1000 = convolution_backward_default_333[1];  convolution_backward_default_333 = None
        le_scalar_119 = torch.ops.aten.le.Scalar(arg1095_1, 0);  arg1095_1 = None
        new_zeros_default_181 = torch.ops.aten.new_zeros.default(getitem_999, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_181 = torch.ops.aten.where.self(le_scalar_119, new_zeros_default_181, getitem_999);  le_scalar_119 = new_zeros_default_181 = getitem_999 = None
        sum_dim_int_list_334 = torch.ops.aten.sum.dim_IntList(where_self_181, [0, 2, 3])
        sub_tensor_579 = torch.ops.aten.sub.Tensor(arg1093_1, arg1907_1);  arg1093_1 = arg1907_1 = None
        mul_tensor_1673 = torch.ops.aten.mul.Tensor(where_self_181, sub_tensor_579)
        sum_dim_int_list_335 = torch.ops.aten.sum.dim_IntList(mul_tensor_1673, [0, 2, 3]);  mul_tensor_1673 = None
        mul_tensor_1674 = torch.ops.aten.mul.Tensor(sum_dim_int_list_334, 0.0005668934240362812);  sum_dim_int_list_334 = None
        view_default_542 = torch.ops.aten.view.default(mul_tensor_1674, [1, 168, 1, 1]);  mul_tensor_1674 = None
        mul_tensor_1675 = torch.ops.aten.mul.Tensor(sum_dim_int_list_335, 0.0005668934240362812)
        mul_tensor_1676 = torch.ops.aten.mul.Tensor(arg1094_1, arg1094_1)
        mul_tensor_1677 = torch.ops.aten.mul.Tensor(mul_tensor_1675, mul_tensor_1676);  mul_tensor_1675 = mul_tensor_1676 = None
        view_default_543 = torch.ops.aten.view.default(mul_tensor_1677, [1, 168, 1, 1]);  mul_tensor_1677 = None
        mul_tensor_1678 = torch.ops.aten.mul.Tensor(arg1094_1, arg253_1);  arg253_1 = None
        view_default_544 = torch.ops.aten.view.default(mul_tensor_1678, [1, 168, 1, 1]);  mul_tensor_1678 = None
        mul_tensor_1679 = torch.ops.aten.mul.Tensor(sub_tensor_579, view_default_543);  sub_tensor_579 = view_default_543 = None
        sub_tensor_580 = torch.ops.aten.sub.Tensor(where_self_181, mul_tensor_1679);  where_self_181 = mul_tensor_1679 = None
        sub_tensor_581 = torch.ops.aten.sub.Tensor(sub_tensor_580, view_default_542);  sub_tensor_580 = view_default_542 = None
        mul_tensor_1680 = torch.ops.aten.mul.Tensor(sub_tensor_581, view_default_544);  sub_tensor_581 = view_default_544 = None
        mul_tensor_1681 = torch.ops.aten.mul.Tensor(sum_dim_int_list_335, arg1094_1);  sum_dim_int_list_335 = arg1094_1 = None
        convolution_backward_default_334 = torch.ops.aten.convolution_backward.default(mul_tensor_1680, arg1092_1, arg252_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1680 = arg1092_1 = arg252_1 = None
        getitem_1002 = convolution_backward_default_334[0]
        getitem_1003 = convolution_backward_default_334[1];  convolution_backward_default_334 = None
        convolution_backward_default_335 = torch.ops.aten.convolution_backward.default(getitem_1002, relu_default_10, arg251_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_1002 = arg251_1 = None
        getitem_1005 = convolution_backward_default_335[0]
        getitem_1006 = convolution_backward_default_335[1];  convolution_backward_default_335 = None
        le_scalar_120 = torch.ops.aten.le.Scalar(relu_default_10, 0);  relu_default_10 = None
        new_zeros_default_182 = torch.ops.aten.new_zeros.default(getitem_1005, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_182 = torch.ops.aten.where.self(le_scalar_120, new_zeros_default_182, getitem_1005);  le_scalar_120 = new_zeros_default_182 = getitem_1005 = None
        add_tensor_183 = torch.ops.aten.add.Tensor(slice_tensor_95, where_self_182);  slice_tensor_95 = where_self_182 = None
        avg_pool2d_backward_default_34 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_94, add_tensor_8, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_94 = add_tensor_8 = None
        add_tensor_184 = torch.ops.aten.add.Tensor(slice_tensor_90, avg_pool2d_backward_default_34);  slice_tensor_90 = None
        add_tensor_185 = torch.ops.aten.add.Tensor(add_tensor_184, avg_pool2d_backward_default_34);  add_tensor_184 = avg_pool2d_backward_default_34 = None
        add_tensor_186 = torch.ops.aten.add.Tensor(add_tensor_185, slice_tensor_93);  add_tensor_185 = None
        avg_pool2d_backward_default_35 = torch.ops.aten.avg_pool2d_backward.default(slice_tensor_93, add_tensor_9, [3, 3], [1, 1], [1, 1], False, False, None);  slice_tensor_93 = add_tensor_9 = None
        add_tensor_187 = torch.ops.aten.add.Tensor(add_tensor_183, avg_pool2d_backward_default_35);  add_tensor_183 = avg_pool2d_backward_default_35 = None
        sum_dim_int_list_336 = torch.ops.aten.sum.dim_IntList(slice_tensor_92, [0, 2, 3])
        sub_tensor_582 = torch.ops.aten.sub.Tensor(arg1090_1, arg1908_1);  arg1090_1 = arg1908_1 = None
        mul_tensor_1682 = torch.ops.aten.mul.Tensor(slice_tensor_92, sub_tensor_582)
        sum_dim_int_list_337 = torch.ops.aten.sum.dim_IntList(mul_tensor_1682, [0, 2, 3]);  mul_tensor_1682 = None
        mul_tensor_1683 = torch.ops.aten.mul.Tensor(sum_dim_int_list_336, 0.0005668934240362812);  sum_dim_int_list_336 = None
        view_default_545 = torch.ops.aten.view.default(mul_tensor_1683, [1, 168, 1, 1]);  mul_tensor_1683 = None
        mul_tensor_1684 = torch.ops.aten.mul.Tensor(sum_dim_int_list_337, 0.0005668934240362812)
        mul_tensor_1685 = torch.ops.aten.mul.Tensor(arg1091_1, arg1091_1)
        mul_tensor_1686 = torch.ops.aten.mul.Tensor(mul_tensor_1684, mul_tensor_1685);  mul_tensor_1684 = mul_tensor_1685 = None
        view_default_546 = torch.ops.aten.view.default(mul_tensor_1686, [1, 168, 1, 1]);  mul_tensor_1686 = None
        mul_tensor_1687 = torch.ops.aten.mul.Tensor(arg1091_1, arg250_1);  arg250_1 = None
        view_default_547 = torch.ops.aten.view.default(mul_tensor_1687, [1, 168, 1, 1]);  mul_tensor_1687 = None
        mul_tensor_1688 = torch.ops.aten.mul.Tensor(sub_tensor_582, view_default_546);  sub_tensor_582 = view_default_546 = None
        sub_tensor_583 = torch.ops.aten.sub.Tensor(slice_tensor_92, mul_tensor_1688);  mul_tensor_1688 = None
        sub_tensor_584 = torch.ops.aten.sub.Tensor(sub_tensor_583, view_default_545);  sub_tensor_583 = None
        mul_tensor_1689 = torch.ops.aten.mul.Tensor(sub_tensor_584, view_default_547);  sub_tensor_584 = view_default_547 = None
        mul_tensor_1690 = torch.ops.aten.mul.Tensor(sum_dim_int_list_337, arg1091_1);  sum_dim_int_list_337 = arg1091_1 = None
        convolution_backward_default_336 = torch.ops.aten.convolution_backward.default(mul_tensor_1689, arg1089_1, arg249_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1689 = arg1089_1 = arg249_1 = None
        getitem_1008 = convolution_backward_default_336[0]
        getitem_1009 = convolution_backward_default_336[1];  convolution_backward_default_336 = None
        convolution_backward_default_337 = torch.ops.aten.convolution_backward.default(getitem_1008, arg1088_1, arg248_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_1008 = arg248_1 = None
        getitem_1011 = convolution_backward_default_337[0]
        getitem_1012 = convolution_backward_default_337[1];  convolution_backward_default_337 = None
        le_scalar_121 = torch.ops.aten.le.Scalar(arg1088_1, 0);  arg1088_1 = None
        new_zeros_default_183 = torch.ops.aten.new_zeros.default(getitem_1011, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_183 = torch.ops.aten.where.self(le_scalar_121, new_zeros_default_183, getitem_1011);  le_scalar_121 = new_zeros_default_183 = getitem_1011 = None
        sum_dim_int_list_338 = torch.ops.aten.sum.dim_IntList(where_self_183, [0, 2, 3])
        sub_tensor_585 = torch.ops.aten.sub.Tensor(arg1086_1, arg1909_1);  arg1086_1 = arg1909_1 = None
        mul_tensor_1691 = torch.ops.aten.mul.Tensor(where_self_183, sub_tensor_585)
        sum_dim_int_list_339 = torch.ops.aten.sum.dim_IntList(mul_tensor_1691, [0, 2, 3]);  mul_tensor_1691 = None
        mul_tensor_1692 = torch.ops.aten.mul.Tensor(sum_dim_int_list_338, 0.0005668934240362812);  sum_dim_int_list_338 = None
        view_default_548 = torch.ops.aten.view.default(mul_tensor_1692, [1, 168, 1, 1]);  mul_tensor_1692 = None
        mul_tensor_1693 = torch.ops.aten.mul.Tensor(sum_dim_int_list_339, 0.0005668934240362812)
        mul_tensor_1694 = torch.ops.aten.mul.Tensor(arg1087_1, arg1087_1)
        mul_tensor_1695 = torch.ops.aten.mul.Tensor(mul_tensor_1693, mul_tensor_1694);  mul_tensor_1693 = mul_tensor_1694 = None
        view_default_549 = torch.ops.aten.view.default(mul_tensor_1695, [1, 168, 1, 1]);  mul_tensor_1695 = None
        mul_tensor_1696 = torch.ops.aten.mul.Tensor(arg1087_1, arg247_1);  arg247_1 = None
        view_default_550 = torch.ops.aten.view.default(mul_tensor_1696, [1, 168, 1, 1]);  mul_tensor_1696 = None
        mul_tensor_1697 = torch.ops.aten.mul.Tensor(sub_tensor_585, view_default_549);  sub_tensor_585 = view_default_549 = None
        sub_tensor_586 = torch.ops.aten.sub.Tensor(where_self_183, mul_tensor_1697);  where_self_183 = mul_tensor_1697 = None
        sub_tensor_587 = torch.ops.aten.sub.Tensor(sub_tensor_586, view_default_548);  sub_tensor_586 = view_default_548 = None
        mul_tensor_1698 = torch.ops.aten.mul.Tensor(sub_tensor_587, view_default_550);  sub_tensor_587 = view_default_550 = None
        mul_tensor_1699 = torch.ops.aten.mul.Tensor(sum_dim_int_list_339, arg1087_1);  sum_dim_int_list_339 = arg1087_1 = None
        convolution_backward_default_338 = torch.ops.aten.convolution_backward.default(mul_tensor_1698, arg1085_1, arg246_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1698 = arg1085_1 = arg246_1 = None
        getitem_1014 = convolution_backward_default_338[0]
        getitem_1015 = convolution_backward_default_338[1];  convolution_backward_default_338 = None
        convolution_backward_default_339 = torch.ops.aten.convolution_backward.default(getitem_1014, relu_default_11, arg245_1, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_1014 = arg245_1 = None
        getitem_1017 = convolution_backward_default_339[0]
        getitem_1018 = convolution_backward_default_339[1];  convolution_backward_default_339 = None
        le_scalar_122 = torch.ops.aten.le.Scalar(relu_default_11, 0);  relu_default_11 = None
        new_zeros_default_184 = torch.ops.aten.new_zeros.default(getitem_1017, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_184 = torch.ops.aten.where.self(le_scalar_122, new_zeros_default_184, getitem_1017);  le_scalar_122 = new_zeros_default_184 = getitem_1017 = None
        add_tensor_188 = torch.ops.aten.add.Tensor(add_tensor_186, where_self_184);  add_tensor_186 = where_self_184 = None
        sub_tensor_588 = torch.ops.aten.sub.Tensor(arg1083_1, arg1910_1);  arg1083_1 = arg1910_1 = None
        mul_tensor_1700 = torch.ops.aten.mul.Tensor(slice_tensor_92, sub_tensor_588)
        sum_dim_int_list_340 = torch.ops.aten.sum.dim_IntList(mul_tensor_1700, [0, 2, 3]);  mul_tensor_1700 = None
        mul_tensor_1701 = torch.ops.aten.mul.Tensor(sum_dim_int_list_340, 0.0005668934240362812)
        mul_tensor_1702 = torch.ops.aten.mul.Tensor(arg1084_1, arg1084_1)
        mul_tensor_1703 = torch.ops.aten.mul.Tensor(mul_tensor_1701, mul_tensor_1702);  mul_tensor_1701 = mul_tensor_1702 = None
        view_default_551 = torch.ops.aten.view.default(mul_tensor_1703, [1, 168, 1, 1]);  mul_tensor_1703 = None
        mul_tensor_1704 = torch.ops.aten.mul.Tensor(arg1084_1, arg244_1);  arg244_1 = None
        view_default_552 = torch.ops.aten.view.default(mul_tensor_1704, [1, 168, 1, 1]);  mul_tensor_1704 = None
        mul_tensor_1705 = torch.ops.aten.mul.Tensor(sub_tensor_588, view_default_551);  sub_tensor_588 = view_default_551 = None
        sub_tensor_589 = torch.ops.aten.sub.Tensor(slice_tensor_92, mul_tensor_1705);  slice_tensor_92 = mul_tensor_1705 = None
        sub_tensor_590 = torch.ops.aten.sub.Tensor(sub_tensor_589, view_default_545);  sub_tensor_589 = view_default_545 = None
        mul_tensor_1706 = torch.ops.aten.mul.Tensor(sub_tensor_590, view_default_552);  sub_tensor_590 = view_default_552 = None
        mul_tensor_1707 = torch.ops.aten.mul.Tensor(sum_dim_int_list_340, arg1084_1);  sum_dim_int_list_340 = arg1084_1 = None
        convolution_backward_default_340 = torch.ops.aten.convolution_backward.default(mul_tensor_1706, arg1082_1, arg243_1, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_tensor_1706 = arg1082_1 = arg243_1 = None
        getitem_1020 = convolution_backward_default_340[0]
        getitem_1021 = convolution_backward_default_340[1];  convolution_backward_default_340 = None
        convolution_backward_default_341 = torch.ops.aten.convolution_backward.default(getitem_1020, arg1081_1, arg242_1, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False]);  getitem_1020 = arg242_1 = None
        getitem_1023 = convolution_backward_default_341[0]
        getitem_1024 = convolution_backward_default_341[1];  convolution_backward_default_341 = None
        le_scalar_123 = torch.ops.aten.le.Scalar(arg1081_1, 0);  arg1081_1 = None
        new_zeros_default_185 = torch.ops.aten.new_zeros.default(getitem_1023, [], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_185 = torch.ops.aten.where.self(le_scalar_123, new_zeros_default_185, getitem_1023);  le_scalar_123 = new_zeros_default_185 = getitem_1023 = None
        sum_dim_int_list_341 = torch.ops.aten.sum.dim_IntList(where_self_185, [0, 2, 3])
        sub_tensor_591 = torch.ops.aten.sub.Tensor(arg1079_1, arg1911_1);  arg1079_1 = arg1911_1 = None
        mul_tensor_1708 = torch.ops.aten.mul.Tensor(where_self_185, sub_tensor_591);  where_self_185 = sub_tensor_591 = None
        sum_dim_int_list_342 = torch.ops.aten.sum.dim_IntList(mul_tensor_1708, [0, 2, 3]);  mul_tensor_1708 = None
        mul_tensor_1709 = torch.ops.aten.mul.Tensor(sum_dim_int_list_341, 0.0005668934240362812);  sum_dim_int_list_341 = None
        view_default_553 = torch.ops.aten.view.default(mul_tensor_1709, [1, 168, 1, 1]);  mul_tensor_1709 = None
        mul_tensor_1710 = torch.ops.aten.mul.Tensor(sum_dim_int_list_342, 0.0005668934240362812);  sum_dim_int_list_342 = None
        mul_tensor_1711 = torch.ops.aten.mul.Tensor(arg1080_1, arg1080_1)
        mul_tensor_1712 = torch.ops.aten.mul.Tensor(mul_tensor_1710, mul_tensor_1711);  mul_tensor_1710 = mul_tensor_1711 = None
        view_default_554 = torch.ops.aten.view.default(mul_tensor_1712, [1, 168, 1, 1]);  mul_tensor_1712 = None
        mul_tensor_1713 = torch.ops.aten.mul.Tensor(arg1080_1, arg241_1);  arg1080_1 = arg241_1 = None
        return (mul_tensor_1713,)
    
args = [((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((336, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((672, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((672, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((168, 168, 1, 1), (168, 1, 1, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((336, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, 'cuda'), ((168, 1008, 1, 1), (1008, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1344, 1, 1), (1344, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1344, 1, 1), (1344, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((336, 336, 1, 1), (336, 1, 1, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((672, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((672, 2016, 1, 1), (2016, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((672, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((672, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((672, 1, 5, 5), (25, 25, 5, 1), torch.float32, 'cuda'), ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((672, 1, 3, 3), (9, 9, 3, 1), torch.float32, 'cuda'), ((672, 672, 1, 1), (672, 1, 1, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 1008, 42, 42), (1778112, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 1008, 42, 42), (1778112, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 1008, 42, 42), (1778112, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 1008, 42, 42), (1778112, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 336, 42, 42), (592704, 1764, 42, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 1008, 42, 42), (1778112, 1764, 42, 1), torch.float32, 'cuda'), ((1, 336, 42, 42), (592704, 1764, 42, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 45, 45), (680400, 2025, 45, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 47, 47), (742224, 2209, 47, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 43, 43), (621264, 1849, 43, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.int64, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 43, 43), (621264, 1849, 43, 1), torch.float32, 'cuda'), ((1, 336, 45, 45), (680400, 2025, 45, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 1008, 21, 21), (444528, 441, 21, 1), torch.float32, 'cuda'), ((1, 1008, 42, 42), (1778112, 1764, 42, 1), torch.float32, 'cuda'), ((1, 1008, 21, 21), (444528, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 1344, 21, 21), (592704, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 2016, 21, 21), (889056, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 2016, 21, 21), (889056, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 2016, 21, 21), (889056, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 2016, 21, 21), (889056, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 2016, 21, 21), (889056, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 672, 21, 21), (296352, 441, 21, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 2016, 21, 21), (889056, 441, 21, 1), torch.float32, 'cuda'), ((1, 672, 21, 21), (296352, 441, 21, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 25, 25), (420000, 625, 25, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 27, 27), (489888, 729, 27, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 23, 23), (355488, 529, 23, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.int64, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 23, 23), (355488, 529, 23, 1), torch.float32, 'cuda'), ((1, 672, 25, 25), (420000, 625, 25, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 672, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 672, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 672, 21, 21), (296352, 441, 21, 1), torch.bool, 'cuda'), ((1, 672, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 672, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 672, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 672, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 672, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 672, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 672, 21, 21), (296352, 441, 21, 1), torch.bool, 'cuda'), ((1, 672, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 672, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 42, 42), (592704, 1764, 42, 1), torch.bool, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 42, 42), (592704, 1764, 42, 1), torch.bool, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 336, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 1, 1), (0, 1, 0, 0), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((168,), (1,), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 168, 42, 42), (296352, 1764, 42, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((336,), (1,), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 336, 21, 21), (148176, 441, 21, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda'), ((1, 2016, 21, 21), (889056, 441, 21, 1), torch.bool, 'cuda'), ((1, 2016, 21, 21), (889056, 441, 21, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (325248, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (325248, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (325248, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 11, 11), (325248, 121, 11, 1), torch.float32, 'cuda'), ((1, 672, 21, 21), (296352, 441, 21, 1), torch.float32, 'cuda'), ((672,), (1,), torch.float32, 'cuda'), ((1, 672, 11, 11), (81312, 121, 11, 1), torch.float32, 'cuda')]
args = [rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args]
mod = make_fx(Repro())(*args)

compiled = compile_fx_inner(mod, args)
compiled(*args)
