import string
import re
import pickle
import random



# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# split a loaded document into sentences
def to_sentences(doc):
	return doc.strip().split('\n')
 
# =============================================================================
# # clean a list of lines
# def clean_lines(lines):
# 	cleaned = list()
# 	# prepare regex for char filtering
# 	re_print = re.compile('[^%s]' % re.escape(string.printable))
# 	# prepare translation table for removing punctuation
# 	table = str.maketrans('', '', string.punctuation)
# 	for line in lines:
# 		# normalize unicode characters
# 		line = normalize('NFD', line).encode('ascii', 'ignore')
# 		line = line.decode('UTF-8')
# 		# tokenize on white space
# 		line = line.split()
# 		# convert to lower case
# 		line = [word.lower() for word in line]
# 		# remove punctuation from each token
# 		line = [word.translate(table) for word in line]
# 		# remove non-printable chars form each token
# 		line = [re_print.sub('', w) for w in line]
# 		# remove tokens with numbers in them
# 		line = [word for word in line if word.isalpha()]
# 		# store as string
# 		cleaned.append(' '.join(line))
# 	return cleaned
#  
# 
# =============================================================================
def clean_text(lines):
    '''Clean text by removing unnecessary characters and altering the format of words.'''
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    cleaned = list()
    for text in lines:
        text = text.lower()
        
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,']", "", text)
                         
        text = text.split()
        text = [re_print.sub('', w) for w in text]
        
        cleaned.append(' '.join(text))
                         
    return cleaned



 
# load English data
filename = 'europarl-v7.fr-en.en'
doc = load_doc(filename)
sentences = to_sentences(doc)
en_sentences = clean_text(sentences)
# spot check
for i in range(10):
	print(en_sentences[i])
 
# load French data
filename = 'europarl-v7.fr-en.fr'
doc = load_doc(filename)
sentences = to_sentences(doc)
fr_sentences = clean_text(sentences)

for i in range(5):
    print(fr_sentences[i])
    print(en_sentences[i])
    
    print('---------')
    
    
min_len = 2
max_len = 20

  
short_eng_temp = []
short_frn_temp = []

for i, text in enumerate(en_sentences):
    if(len(text.split()) >= min_len and len(text.split())<= max_len):
        short_eng_temp.append(text)
        short_frn_temp.append(fr_sentences[i])

short_eng = []
short_frn = []

for i, text in enumerate(short_frn_temp):
    if(len(text.split()) >= min_len and len(text.split())<= max_len):
        short_frn.append(text)
        short_eng.append(short_eng_temp[i])


for i in range(5):
    print(short_eng[i])
    print(short_frn[i])
    
    print('---------')



eng_vocab = {}
frn_vocab = {}

for text in short_eng:
    for word in text.split():
        if word not in eng_vocab:
            eng_vocab[word] = 1
        else:
            eng_vocab[word] += 1

for text in short_frn:
    for word in text.split():
        if word not in frn_vocab:
            frn_vocab[word] = 1
        else:
            frn_vocab[word] += 1
            
            
thresh = 2


eng_vocab_to_int = {}
word_num = 0
for word, count in eng_vocab.items():
    if count >= thresh:
        eng_vocab_to_int[word] = word_num
        word_num += 1


frn_vocab_to_int = {}
word_num = 0
for word, count in frn_vocab.items():
    if count >= thresh:
        frn_vocab_to_int[word] = word_num
        word_num += 1
        


codes = ['<PAD>','<EOS>','<UNK>','<GO>']

for code in codes:
    eng_vocab_to_int[code] = len(eng_vocab_to_int)+1
    
for code in codes:
    frn_vocab_to_int[code] = len(frn_vocab_to_int)+1
    

eng_int_to_vocab = {num: word for word, num in eng_vocab_to_int.items()}
frn_int_to_vocab = {num: word for word, num in frn_vocab_to_int.items()}

for i in range(len(short_frn)):
    short_frn[i] += ' <EOS>'



eng_ints = []
for line in short_eng:
    ints = []
    for word in line.split():
        if word not in eng_vocab_to_int:
            ints.append(eng_vocab_to_int['<UNK>'])
        else:
            ints.append(eng_vocab_to_int[word])
    eng_ints.append(ints)


frn_ints = []
for line in short_frn:
    ints = []
    for word in line.split():
        if word not in frn_vocab_to_int:
            ints.append(frn_vocab_to_int['<UNK>'])
        else:
            ints.append(frn_vocab_to_int[word])
    frn_ints.append(ints)


# =============================================================================
# combined = list(zip(eng_ints, frn_ints))
# random.shuffle(combined)
# sf_eng_ints, sf_frn_ints = zip(*combined)
# 
# less_eng_ints = sf_eng_ints[0:258452]
# less_frn_ints = sf_frn_ints[0:258452]
# 
# eng_ints = less_eng_ints
# frn_ints = less_frn_ints
# 
# =============================================================================

sorted_eng = []
sorted_frn = []

for length in range(1, max_len+1):
    for i in enumerate(eng_ints):
        if len(i[1]) == length:
            sorted_eng.append(eng_ints[i[0]])
            sorted_frn.append(frn_ints[i[0]])


for i in range(5,10):
    
    for word in sorted_eng[i]:
        print(eng_int_to_vocab[word])
    
    for word in sorted_frn[i]:
        print(frn_int_to_vocab[word])
    
    print('---------')



pickle.dump(((sorted_eng, sorted_frn), (eng_int_to_vocab, frn_int_to_vocab), (eng_vocab_to_int, frn_vocab_to_int)),
            open('preprocess_en_fr.p', 'wb'))




































