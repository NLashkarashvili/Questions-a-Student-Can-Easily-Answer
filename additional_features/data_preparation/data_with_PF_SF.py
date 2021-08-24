#DATA PREPARATION FOR DATA
#WHEN PRACTICE AND STUDENT FEATURES
#ARE PASSED TO THE MODEL ALONGSIDE
#QUES_ID, CHAPTER_ID, SUB_CHAPTER_ID
import numpy as np
import pandas as pd
import warnings
import gc


# load the data
import pandas as pd
dataframe = pd.read_csv('../input/practice-log-demographics/Practice_Log_Demographics.csv', low_memory=False)

#sort data based on timestamp
dataframe = dataframe.sort_values(by=['start_practice'])

FEATURES = ['chapter_label', 'sub_chapter_label','question_name','user_id','term', 'STDNT_SEX_CD', 
            'NonNativeEnglish',  'White','Asian','WhiteOrAsian','Hispanic', 'AfricanAmerican','OtherEthnicities',
            'NonWhiteOrAsian',  'STDNT_CTZN_STAT_CD', 'international', 'gradingType','birthYear', 'exclClassCumGPA',
            'Freshman', 'Junior',   'Sophomore', 'Senior', 'termCreditsGPA', 'termCreditsNoGPA', 'athlete_1',
            'honorsPro', 'LSA', 'programBusiness', 'programEngineering', 'programInformation', 'programOther',
            'HSCalculusTaken', 'highSchoolGPA',  'majorsCount', 'minorsCount', 'PREV_TERM_CUM_GPA',
            'classGraded', 'classHonors',  'Pass_Fail', 'parentsGraduateEdu',  'minorityGroup', 'q',
            'available_flashcards', 'start_practice',  'end_practice', 'days_offset']

dataframe['available_flashcards'] = dataframe["day's_available_flashcards"][:]
dataframe = dataframe.drop(["day's_available_flashcards"], axis=1)
dataframe['user_id'] = dataframe["user_id.x"][:]
dataframe = dataframe.drop(["user_id.x"], axis=1)
dataframe = dataframe[FEATURES]

#encoding term, user_id, chapter_label, sub_chapter_label, question_name
dataframe['term'] = dataframe['term'].astype('category')
dataframe['user_id'] = dataframe['user_id'].astype(int)
dataframe['user_id'] = dataframe['user_id'].astype(str)
dataframe['user_id'] = dataframe['term'].str.cat(dataframe['user_id'], sep=':')
dataframe['user_id'] = dataframe['user_id'].astype('category')
dataframe['chapter_label'] = dataframe['chapter_label'].astype('category')
dataframe['sub_chapter_label'] = dataframe['sub_chapter_label'].astype('category')
dataframe['question_name'] = dataframe['question_name'].astype('category')


dataframe.start_practice = pd.to_datetime(dataframe.start_practice, format='%Y-%m-%d %H:%M:%S')
dataframe.end_practice = pd.to_datetime(dataframe.end_practice, format='%Y-%m-%d %H:%M:%S')
dataframe['dif'] = dataframe.end_practice - dataframe.start_practice
dataframe['dif'] = dataframe['dif'] /np.timedelta64(1, 's')
dataframe['answer_correct'] = np.where((dataframe['q']==5) & (dataframe['dif'] <= 60), 1, 0)


#drop column end_practice
dataframe.drop(columns=['end_practice'], inplace=True)

# calculate the age feature
dataframe['term_value'] = [int(ele[3:]) for ele in dataframe['term']]
dataframe['age'] = dataframe['term_value'] - dataframe['birthYear']

# drop term_value and birthYear column
dataframe.drop(columns=['term_value', 'birthYear'], inplace=True)

# convert minors_count to int value
new_minors_count = []
for i in dataframe['minorsCount']:
    if i == 0 or i == '0':
        new_minors_count.append(0)
    elif i == '1 Minor':
        new_minors_count.append(1)
    else:
        new_minors_count.append(2)

dataframe['minorsCount'] = new_minors_count


for category in ['term','chapter_label', 'sub_chapter_label', 'question_name']:
    dataframe[category] =  dataframe[category].cat.codes



NUMERIC_FEATURE =  ['age',
            'exclClassCumGPA',
            'termCreditsGPA',
            'termCreditsNoGPA',
            'highSchoolGPA', 
            'majorsCount', 'minorsCount',
            'PREV_TERM_CUM_GPA',
            'available_flashcards', 
            'days_offset']
# z-score normalize the numerical features
for f in NUMERIC_FEATURE:
    m = dataframe[f].mean()
    std = dataframe[f].std()
    dataframe[f] = (dataframe[f] - m)/std

grouped_data = dataframe[FEATURE_TRANS].groupby(['user_id']).apply(lambda r: (
                r['answer_correct'],
                r['term'],
                r['chapter_label'],
                r['sub_chapter_label'],
                r['question_name'],
                np.array([
                    r['STDNT_SEX_CD'],r['STDNT_CTZN_STAT_CD'], r['international'], 
                  r['White'],r['Asian'],r['NonWhiteOrAsian'],
                 r['age'],r['exclClassCumGPA'],
                r['Freshman'], r['Junior'], r['Sophomore'], r['Senior'],
                r['termCreditsGPA'], r['termCreditsNoGPA'],
                r['athlete_1'], r['honorsPro'],
                r['LSA'], r['programBusiness'], r['programEngineering'], 
                r['programInformation'], r['programOther'],
                r['HSCalculusTaken'],  r['highSchoolGPA'], 
                r['majorsCount'], r['minorsCount'],
                r['PREV_TERM_CUM_GPA'], 
                r['parentsGraduateEdu'], r['minorityGroup'],
                r['available_flashcards'],
                r['days_offset']
              ]).transpose()
                ))


# remove students who don't have make any interactions with the tool
toRemove = []
for index in grouped_data.index:
    if len(grouped_data[index][0]) <= 10:
        toRemove.append(index)
grouped_data = grouped_data.drop(index=toRemove)





#create dataset class
#to prepare it for train, valid, and test sets
#for additional features we set it to following values
#13 -> for rnn_based models and baseline logistic
#400 -> for transformer_based and basline logistc
from torch.utils.data import Dataset, DataLoader
class SPACE_DATASET(Dataset):
    def __init__(self, data, maxlength = 13):
        super(SPACE_DATASET, self).__init__()
        self.maxlength = maxlength
        self.data = data
        self.users = list()
        for user in data.index:
            self.users.append(user)

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, ix):
        user = self.users[ix]
        user = user
        target, term, ch_label, sub_ch_label, ques_name, features = self.data[user]
        
        ori_target = target.values 
        ch_label = ch_label.values + 1
        sub_ch_label = sub_ch_label.values +1
        term = term.values
        # one hot for term
        term_encode = [0]*7
        term_encode[term[0]] = 1
        ques_name = ques_name.values + 1
        
        
        n = len(ch_label)
        
        # get  user interaction informations in the previous MAXLEN interactions
        if n > self.maxlength:
            ch_label = ch_label[-self.maxlength:]
            sub_ch_label = sub_ch_label[-self.maxlength:]
            ques_name = ques_name[-self.maxlength:]
            target = ori_target[-self.maxlength:]
            shifted_target = ori_target[ (-self.maxlength - 1) :-1]

        else:
            ch_label = [0]*(self.maxlength - n)+list(ch_label[:])
            sub_ch_label = [0]*(self.maxlength - n)+list(sub_ch_label[:])
            ques_name = [0]*(self.maxlength - n)+list(ques_name[:])
            target = [-1]*(self.maxlength - n) + list(ori_target[:])
            shifted_target = [2]*(self.maxlength + 1 - n) + list(ori_target[:-1])

        new_features = []
        count = 0
        for f in features:
            temp = list(f)
            temp.extend(term_encode)
            new_features.append(temp)
            count += 1
            
        features = new_features
        
        return ch_label,sub_ch_label,ques_name,features, shifted_target, target