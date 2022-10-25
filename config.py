MAX_SEQ_LEN=256 # 最大句子长度
TAG_LEN=33  # 事件类型数
# 数据集路径
TRAIN_PATH = './data/ace_ed_txt/train.csv'
DEV_PATH = './data/ace_ed_txt/dev.csv'
TEST_PATH = './data/ace_ed_txt/test.csv'
# ace2005数据集的标签
event_type_to_id= {'Attack':0, 'Die':1, 'Transfer-Ownership':2, 'Start-Position':3, 'Transfer-Money':4, 'Marry':5, 'Start-Org':6, 'Pardon':7, 'Sue':8, 'Divorce':9, 'Demonstrate':10, 'Be-Born':11, 'Acquit':12, 'Charge-Indict':13, 'Release-Parole':14, 'Phone-Write':15, 'Merge-Org':16, 'Elect':17, 'Convict':18, 'Declare-Bankruptcy':19, 'Fine':20, 'Meet':21, 'Sentence':22, 'Nominate':23, 'Injure':24, 'Appeal':25, 'Extradite':26, 'Arrest-Jail':27, 'End-Org':28, 'Trial-Hearing':29, 'Execute':30, 'End-Position':31, 'Transport':32}
id_to_event_type={0:'Attack',1:'Die',2:'Transfer-Ownership',3:'Start-Position',4:'Transfer-Money',5: 'Marry',6:'Start-Org',7:'Pardon',8:'Sue',9:'Divorce',10:'Demonstrate',11:'Be-Born',12:'Acquit',13:'Charge-Indict',14:'Release-Parole',15:'Phone-Write',16:'Merge-Org',17:'Elect',18:'Convict',19:'Declare-Bankruptcy',20:'Fine',21:'Meet',22:'Sentence',23:'Nominate',24: 'Injure',25:'Appeal',26:'Extradite',27:'Arrest-Jail',28:'End-Org',29:'Trial-Hearing',30:'Execute',31:'End-Position',32: 'Transport'}
