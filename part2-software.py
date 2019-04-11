
import pandas as pd
import string
import csv
import random
import math


cd C:\Users\theo\Desktop\part_2\dataset


data = pd.DataFrame.from_csv('261K_lyrics_from_MetroLyrics.csv', sep=",",index_col='ID',header=0)


def normalize(words): # function that removes punctuation and converts to lowercase
    new_words=words.translate(str.maketrans('', '', string.punctuation)).lower()

    return new_words


data.lyrics=data.lyrics.apply(lambda x:normalize(x))


def create_unique_shingles(empty_dict): 
    for i in range(0,len(data)):
        unique_shingles=set()# set because we dont want to take into account the same shingle in the same song
        try: # song number 158 doesnt exist
            tokens=data['lyrics'][i].split()
            if len(tokens) >=3: # there are some songs with fewer words
                for index in range(len(tokens) - 3 + 1):
                    # Construct the shingle text by combining k words together.
                    shingle = tokens[index:index + 3]
                    # Hash the shingle to a 32-bit integer.  
                    shingle = ' '.join(shingle)
                    if shingle not in unique_shingles:
                        unique_shingles.add(shingle)
                    else:
                        del shingle
                        index = index - 1
            
        except:
            continue
        empty_dict[i]=unique_shingles
    return(empty_dict)


results={}
create_unique_shingles(results)

d={}
id = 0
for k, v in results.items():# this function assings an id number for each unique shingle of a song. it is kinda the opposite of what we did in the previosu function, because now the key of this dictionary is the shingle, not the document id, and the value is an id for each shingle.
    for elem in v:
        if str(elem) not in d.keys():
            d[str(elem)]=id
            id += 1


numsh={}

for k, v in results.items(): # this dictionary is practically the tsv that we will create later. it has as key the id of each song and as as values the set of unique shingles of each song.
    lista=set()
    for elem in v:
        number=d.get(str(elem))
        lista.add(number)
    numsh[k]=list(lista)



file=open("prova.tsv", "w",newline='') # create the tsv file with the shingles
w = csv.writer(file, delimiter='\t')
w.writerow(['set_id','set_as_list_of_elements_id'])
for key, val in numsh.items():
    if len(val)>=1:
        #w.writerow([key, val])
        w.writerow(['id_'+str(key), val])
file.close()



################################################
num_hash_functions = 100
upper_bound_on_number_of_distinct_terms  = len(d.keys())
#upper_bound_on_number_of_distinct_terms =   138492
#upper_bound_on_number_of_distinct_terms =  3746518

################################################


### primality checker
def is_prime(number):
    for j in range(2, int(math.sqrt(number)+1)):
        if (number % j) == 0: 
            return False
    return True


file=open("hashtable.tsv", "w",newline='')
w = csv.writer(file, delimiter='\t')
w.writerow(['a', 'b', 'p', 'n'])
set_of_all_hash_functions = set()
while len(set_of_all_hash_functions) < num_hash_functions:
    a = random.randint(1, upper_bound_on_number_of_distinct_terms-1)
    b = random.randint(0, upper_bound_on_number_of_distinct_terms-1)
    p = random.randint(upper_bound_on_number_of_distinct_terms, 10*upper_bound_on_number_of_distinct_terms)
    while is_prime(p) == False:
        p = random.randint(upper_bound_on_number_of_distinct_terms, 10*upper_bound_on_number_of_distinct_terms)
    current_hash_function_id = tuple([a, b, p])
    set_of_all_hash_functions.add(current_hash_function_id)
    w.writerow([a, b, p, upper_bound_on_number_of_distinct_terms])  
file.close()



java -Xmx3G tools.NearDuplicatesDetector lsh_plus_min_hashing 0.88 5 20 ./dataset/hashtable.tsv ./dataset/prova.tsv ./nearduplicate.tsv



def calculate(j):
    print(1-(1-j**5)**20)

