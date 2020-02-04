import csv
import pandas as pd
import numpy as np
import re

def import_files():
    global reviews
    global feature_dictionary_list
    with open('hotelPosT-train.txt') as posf:
        reviews = posf.readlines()
    for line in reviews:
        feature_dictionary = create_dictionary()
        vocab = parse_input(feature_dictionary, line)
        feature_dictionary["class"] = 1
        compute_1(feature_dictionary, vocab)
        compute_2(feature_dictionary, vocab)
        compute_3(feature_dictionary, vocab)
        compute_4(feature_dictionary, vocab)
        compute_5(feature_dictionary, vocab)
        compute_6(feature_dictionary, vocab)
        feature_dictionary_list.append(feature_dictionary)


    with open('hotelNegT-train.txt') as negf:
        reviews = negf.readlines()
        for line in reviews:
            feature_dictionary = create_dictionary()
            vocab = parse_input(feature_dictionary, line)
            feature_dictionary["class"] = 0
            compute_1(feature_dictionary, vocab)
            compute_2(feature_dictionary, vocab)
            compute_3(feature_dictionary, vocab)
            compute_4(feature_dictionary, vocab)
            compute_5(feature_dictionary, vocab)
            compute_6(feature_dictionary, vocab)
            feature_dictionary_list.append(feature_dictionary)



def import_files_testdata(filename):
    global reviews_test_data
    global feature_dictionary_list_test_data
    with open(filename) as f:
        reviews_test_data = f.readlines()
    for line in reviews_test_data:
        feature_dictionary_test_data = create_dictionary_test_data()
        vocab = parse_input(feature_dictionary_test_data, line)
        compute_1(feature_dictionary_test_data, vocab)
        compute_2(feature_dictionary_test_data, vocab)
        compute_3(feature_dictionary_test_data, vocab)
        compute_4(feature_dictionary_test_data, vocab)
        compute_5(feature_dictionary_test_data, vocab)
        compute_6(feature_dictionary_test_data, vocab)
        feature_dictionary_list_test_data.append(feature_dictionary_test_data)


def parse_file(filename):
    content_array=[]
    with open(filename) as f:
        for line in f:
            content_array.append(line)
    content_array = [word.rstrip('\n') for word in content_array]
    return content_array

def parse_input(feature_dictionary,line):
    Id_seperated_review = line.split('\t')
    feature_dictionary["Review-ID"] = Id_seperated_review[0]
    reviews = Id_seperated_review[1].split()
    vocab1 = [word.lower() for word in reviews]
    vocab = [re.sub('[,.]', '', item) for item in vocab1]
    return vocab

def create_dictionary():

    feature_dict = {
        "Review-ID" : 0,
        "positive_lexicon_count": 0,
        "negative_lexicon_count": 0,
        "if_no": 0,
        "count_1st_and_2nd_pronouns": 0,
        "if_!": 0,
        "log(word_count_review)": 0,
        "class":0,
    }
    return feature_dict

def create_dictionary_test_data():

    feature_dict = {
        "Review-ID" : 0,
        "positive_lexicon_count": 0,
        "negative_lexicon_count": 0,
        "if_no": 0,
        "count_1st_and_2nd_pronouns": 0,
        "if_!": 0,
        "log(word_count_review)": 0,
    }
    return feature_dict


def compute_1(feature_dict,vocab):

    for search_word in vocab:
        if search_word in content_array_pos:
            feature_dict["positive_lexicon_count"] += 1

    #print(feature_dict["positive_lexicon_count"])


def compute_2(feature_dict,vocab):

    for search_word in vocab:
        if search_word in content_array_neg:
            feature_dict["negative_lexicon_count"] += 1

    #print(feature_dict["negative_lexicon_count"])

def compute_3(feature_dict,vocab):
    for word in vocab:
        if word in ["no","No","NO"]:
            feature_dict["if_no"] = 1
            break;

    #print(feature_dict["if_no"])


def compute_4(feature_dict,vocab):
    pronoun_list = ["I", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]
    list = [word.lower() for word in pronoun_list]
    for word in vocab:
        if word in list:
            feature_dict["count_1st_and_2nd_pronouns"] += 1

    #print(feature_dict["count_1st_and_2nd_pronouns"])


def compute_5(feature_dict,vocab):
    for word in vocab:
        if word.endswith('!'):
            feature_dict["if_!"] = 1
            break;
    #print(feature_dict["if_!"])

def compute_6(feature_dict,vocab):
    feature_dict["log(word_count_review)"] = round(np.log(len(vocab)),2)
    #print(feature_dict["log(word_count_review)"])


def write_to_csv(feature_dictionary_list):
    header = ["Review-ID", "positive_lexicon_count","negative_lexicon_count","if_no","count_1st_and_2nd_pronouns","if_!","log(word_count_review)","class"]
    with open('prasad-veena-assgn2-part1.csv', 'w') as csvFile:
        writer = csv.DictWriter(
            csvFile, fieldnames= header)
        #writer.writeheader()
        writer.writerows(feature_dictionary_list)
    csvFile.close()

    with open('prasad-veena-assgn2-header-part1.csv', 'w') as csvFile:
        writer = csv.DictWriter(
            csvFile, fieldnames=header)
        writer.writeheader()
        writer.writerows(feature_dictionary_list)
    csvFile.close()


def write_to_csv_test_data(feature_dictionary_list_test_data):
    header = ["Review-ID", "positive_lexicon_count","negative_lexicon_count","if_no","count_1st_and_2nd_pronouns","if_!","log(word_count_review)"]
    with open('prasad-veena-assgn2-part.csv', 'w') as csvFile:
        writer = csv.DictWriter(
            csvFile,fieldnames= header)
        #writer.writeheader()
        writer.writerows(feature_dictionary_list_test_data)
    csvFile.close()

    with open('prasad-veena-assgn2-header-part.csv', 'w') as csvFile:
        writer = csv.DictWriter(
            csvFile,fieldnames= header)
        writer.writeheader()
        writer.writerows(feature_dictionary_list_test_data)
    csvFile.close()

def read_csv_file_df(filename):
    df_csv = pd.read_csv(filename,delimiter = ',')
    pos_review_list = df_csv.loc[df_csv['class'] == 1 ]
    neg_review_list = df_csv.loc[df_csv['class'] == 0 ]
    return pos_review_list,neg_review_list

def read_csv_file_reviewdf(filename):
    reviewdf_csv = pd.read_csv(filename,delimiter = ',')
    return reviewdf_csv

def training_data(df1,df2):
    df180 = df1.sample(frac=.8)
    df280 = df2.sample(frac=.8)
    frames1 = [df180, df280]
    df80 = pd.concat(frames1)
    df80['F7'] = 1
    return df80

def testing_data(df1,df2):
    df120 = df1.sample(frac=.2)
    df220 = df2.sample(frac=.2)
    frames2 = [df120, df220]
    df20 = pd.concat(frames2)
    df20['F7']= 1
    return df20

def select_random_example(df):
     return(df.sample())

def drop_columns(df):
    df = df.drop(columns=['Review-ID', 'class']).astype('float')
    return df

def create_example(df):
    df = df.drop(columns=['Review-ID', 'class']).astype('float')
    return np.squeeze(np.asarray(df))

def classprob(score):
    return 1 / (1 + np.exp(-score))

def calculate_accuracy(count,size):
    accuracy = (count/(size))*100
    return accuracy

def create_array_matrix(df):
    return np.squeeze(np.asarray(df))

def SGD_training(training_df,testing_df,weights):

    while(1):
        example = select_random_example(training_df)
        correct = int(example.iloc[0]['class'])
        example = create_example(example)
        rawscore = np.dot(weights, example)
        score = classprob(rawscore)
        gradient =  (score - correct) * example
        learningrate = 1
        new_weights = weights - learningrate * gradient
        ####SGDTesting
        accuracy = SGD_testing(testing_df, new_weights)
        if (accuracy > 90):
            break
    return new_weights

def SGD_testing(testing_df,new_weights):
    count = 0
    example_matrix = create_array_matrix(testing_df)
    #print(new_weights)
    f = open("result.txt", "w")

    for i in range(len(example_matrix)):
        examples = example_matrix[i]
        correct = int(examples[7])
        review_id = examples[0]
        example = examples[1:-2]
        example = np.append(example,examples[-1])
        new_score = classprob(np.dot(new_weights, example))
        if(new_score > 0.5):
            classifier = 1
            txt = "POS"
        else:
            classifier = 0
            txt = "NEG"

        f.write("{}\t{}\n".format(review_id,txt))
        if(classifier == correct):
            count +=1
    accuracy = calculate_accuracy(count,len(example_matrix))
    print("The Classifier works with the precision {}" .format(accuracy))
    f.close()
    return accuracy

def SGD_testing_test_data(test_data_df,new_weights):
    count = 0
    test_data_df['F7'] = 1
    example_matrix = create_array_matrix(test_data_df)
    print(new_weights)
    f = open("prasad-veena-assgn2-out.txt", "w")
    for i in range(len(example_matrix)):
        examples = example_matrix[i]
        review_id = examples[0]
        example = examples[1:]
        new_score = classprob(np.dot(new_weights, example))
        if(new_score > 0.5):
            classifier = 1
            txt = "POS"
        else:
            classifier = 0
            txt = "NEG"

        f.write("{}\t{}\n".format(review_id,txt))

    f.close()

def feature_extraction_of_training_data():
    import_files()
    global reviews
    global feature_dictionary_list
    write_to_csv(feature_dictionary_list)

def train_test_training_data():
    ###SGDTraining
    global final_weights
    bias = 0.1
    pos_feature_list, neg_feature_list = read_csv_file_df('prasad-veena-assgn2-header-part1.csv')
    training_df = training_data(pos_feature_list, neg_feature_list)
    testing_df = testing_data(pos_feature_list, neg_feature_list)
    initial_weights = np.array([2.5, -5, -1.2, 0.5, 2, 0.7, bias])
    final_weights = SGD_training(training_df, testing_df, initial_weights)

def evaluate_test_data():
    ###Test data
    ###Feature Extraction
    ##Final weights used for to generate the output file for test data is
    ### weights = [ 2.01219646 -5.48780354 -1.36260118 -0.15040471  1.83739882  0.01219701  -0.06260118]

    filename_test_data = "HW2-testset.txt"
    import_files_testdata(filename_test_data)
    global reviews_test_data
    global feature_dictionary_list_test_data
    global final_weights
    write_to_csv_test_data(feature_dictionary_list_test_data)

    ###Evaluate
    filename = 'prasad-veena-assgn2-header-part.csv'
    review_df = read_csv_file_reviewdf(filename)
    SGD_testing_test_data(review_df, final_weights)

def make_choice(argument):
    print("Selected {}".format(argument))
    if(argument == 1):
        feature_extraction_of_training_data()
    if(argument == 2):
         train_test_training_data()
    if(argument == 3):
        ##Final weights used for to generate the output file for test data is
        ##weights = [ 2.01219646 -5.48780354 -1.36260118 -0.15040471  1.83739882  0.01219701  -0.06260118]
        evaluate_test_data()


def main():
    print("python main function")
    print("Enter 1 to Extract Feature of Training Data \n   2 to Train the Training Data\n   3 to Evaluate the Test Data \n   -1 to quit")
    num=0
    while(num != -1):
        num = input("Enter:")
        a = int(num)
        if (a == -1):
            break
        make_choice(int(num))

if __name__ == '__main__':
    global content_array_pos
    global content_array_neg
    reviews = []
    reviews_test_data = []
    feature_dictionary_list = []
    feature_dictionary_list_test_data = []
    final_weights = []
    content_array_pos = parse_file("positive-words.txt")
    content_array_neg = parse_file("negative-words.txt")
    ##Final weights used for to generate the output file for test data is
    ## weights = [ 2.01219646 -5.48780354 -1.36260118 -0.15040471  1.83739882  0.01219701  -0.06260118]
    main()
