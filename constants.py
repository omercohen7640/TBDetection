N_SHEETS = 5
DATASET = "TB_dataset.xlsx"
FEATURES_LIST = ['Rv 3881c','Rv 0934 (P38 or PstS1)','Rv 2031c (HspX)','Rv 1886c (Ag85b)','Rv 1860(MPT32 )','Rv 3874 (CFP10)','H37 Rv MEM','Rv 1926c','Rv 1984c (CFP21)','Rv 3841 (Bfrb1)','Rv 2875 (MPT70)','Rv 3875(ESAT6)','Rv 3804c (Ag85a)','HN878 MEM','CDC 1551 MEM','Rv 3418c(Gro ES)','Rv 3507','Rv 3874-Rv 3875 (CFP10-ESAT6)','Rv 2878c','Rv 1099','Rv 3619','Rv 1677', 'Rv 2220','Rv 2032','Rv 3873','Rv 0054','Rv 1566c','Rv 0129c (Ag85c)','Rv 1009','Rv 1980 (MPT64)','Rv 0831']
TB = 1
NON_TB = 0
COPD = 1
LABELS_OF_SHEETS = [TB, TB, TB, NON_TB, COPD]
# LABELS_OF_SHEETS = [0, 1, 2, 3, 4]
N_SAMPLES = 200
N = 358
GBM_PARAM = {
    "task": "train",
    "objective": "binary"
}
epsilon = 10e3