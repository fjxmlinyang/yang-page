---
title: "Project: Building a Spam Filter with Naive Bayes"
excerpt: "Explore Bayes model <br/>"
collection: projects
---



This is the project from Dataquest

1. Assign probabilities to events based on certain conditions by using condtional probability rules
2. Assign probabilities to events based on whether they are inrelationship of statistical independence or not with other events
3. Assign probabilities to events based on prior knowledge by usin Bayes' theorem
4. Create a spam filter for SMS messages using the multinomial Naive Bayes algorithm.




```python
import pandas as pd

```

Project——OSEMN Pipeline
## Step1 O- Obtaining our data
1. file name: SMSSpamCollection
2. explore a little bit


```python
sms_spam=pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['Label','SMS'])

```


```python
sms_spam.head()
sms_spam.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5572</td>
      <td>5572</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>5169</td>
    </tr>
    <tr>
      <th>top</th>
      <td>ham</td>
      <td>Sorry, I'll call later</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4825</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
sms_spam['Label'].value_counts(normalize= True)
```




    ham     0.865937
    spam    0.134063
    Name: Label, dtype: float64



we read in the dataset and saw that about 87% of the messages are ham ("ham" means non-spam), and the remaining 13% are spam.

### First try: Training and Testing data
1. Split data set as training set(80%) and test set(20%)


```python
# Randomize the dataset
data_randomized=sms_spam.sample(frac=1,random_state=1)

# Calculate index for split
training_test_index = round(len(data_randomized)*0.8)

# Training/ Test split
training_set = data_randomized[: training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set)
print(test_set)

```

         Label                                                SMS
    0      ham                       Yep, by the pretty sculpture
    1      ham      Yes, princess. Are you going to make me moan?
    2      ham                         Welp apparently he retired
    3      ham                                            Havent.
    4      ham  I forgot 2 ask ü all smth.. There's a card on ...
    5      ham  Ok i thk i got it. Then u wan me 2 come now or...
    6      ham  I want kfc its Tuesday. Only buy 2 meals ONLY ...
    7      ham                         No dear i was sleeping :-P
    8      ham                          Ok pa. Nothing problem:-)
    9      ham                    Ill be there on  &lt;#&gt;  ok.
    10     ham  My uncles in Atlanta. Wish you guys a great se...
    11     ham                                           My phone
    12     ham                       Ok which your another number
    13     ham  The greatest test of courage on earth is to be...
    14     ham  Dai what this da.. Can i send my resume to thi...
    15     ham                      I am late. I will be there at
    16    spam  FreeMsg Why haven't you replied to my text? I'...
    17     ham                  K, text me when you're on the way
    18    spam  Congrats! 2 mobile 3G Videophones R yours. cal...
    19     ham  Please leave this topic..sorry for telling that..
    20     ham  Ooooooh I forgot to tell u I can get on yovill...
    21     ham          Hi this is yijue, can i meet u at 11 tmr?
    22     ham  I want to show you the world, princess :) how ...
    23     ham                  Well that must be a pain to catch
    24     ham                Well. You know what i mean. Texting
    25     ham         Your bill at 3 is £33.65 so thats not bad!
    26     ham                       Yeah, where's your class at?
    27     ham                                     What's ur pin?
    28     ham  Fighting with the world is easy, u either win ...
    29     ham  Dude. What's up. How Teresa. Hope you have bee...
    ...    ...                                                ...
    4428   ham                Uncle Abbey! Happy New Year. Abiola
    4429   ham             From tomorrow onwards eve 6 to 3 work.
    4430   ham                      Haha, just what I was thinkin
    4431   ham                           Happy New Year Princess!
    4432   ham  Let me know how to contact you. I've you settl...
    4433  spam  I don't know u and u don't know me. Send CHAT ...
    4434   ham         Yes! How is a pretty lady like you single?
    4435   ham                             Just getting back home
    4436   ham                         Oh my God. I'm almost home
    4437  spam  Congratulations YOU'VE Won. You're a Winner in...
    4438   ham  I cant pick the phone right now. Pls send a me...
    4439  spam  Win the newest “Harry Potter and the Order of ...
    4440   ham                    Dear umma she called me now :-)
    4441   ham                        Aight, lemme know what's up
    4442   ham  Good evening Sir, hope you are having a nice d...
    4443  spam  Someone U know has asked our dating service 2 ...
    4444   ham         2marrow only. Wed at  &lt;#&gt;  to 2 aha.
    4445   ham        Haha yeah I see that now, be there in a sec
    4446   ham                        aathi..where are you dear..
    4447   ham  Pls give her the food preferably pap very slow...
    4448   ham             I donno its in your genes or something
    4449  spam  YOUR CHANCE TO BE ON A REALITY FANTASY SHOW ca...
    4450   ham                             Prakesh is there know.
    4451   ham  The beauty of life is in next second.. which h...
    4452   ham             How about clothes, jewelry, and trips?
    4453   ham  Sorry, I'll call later in meeting any thing re...
    4454   ham  Babe! I fucking love you too !! You know? Fuck...
    4455  spam  U've been selected to stay in 1 of 250 top Bri...
    4456   ham  Hello my boytoy ... Geeee I miss you already a...
    4457   ham                           Wherre's my boytoy ? :-(
    
    [4458 rows x 2 columns]
         Label                                                SMS
    0      ham          Later i guess. I needa do mcat study too.
    1      ham             But i haf enuff space got like 4 mb...
    2     spam  Had your mobile 10 mths? Update to latest Oran...
    3      ham  All sounds good. Fingers . Makes it difficult ...
    4      ham  All done, all handed in. Don't know if mega sh...
    5      ham  But my family not responding for anything. Now...
    6      ham                                           U too...
    7      ham  Boo what time u get out? U were supposed to ta...
    8      ham  Genius what's up. How your brother. Pls send h...
    9      ham                             I liked the new mobile
    10     ham                          For my family happiness..
    11     ham  If i let you do this, i want you in the house ...
    12     ham  Do you know why god created gap between your f...
    13     ham  K and you're sure I don't have to have consent...
    14     ham                                    Try neva mate!!
    15     ham  Haha... They cant what... At the most tmr forf...
    16     ham  Doc prescribed me morphine cause the other pai...
    17     ham  Spending new years with my brother and his fam...
    18     ham           and  picking them up from various points
    19    spam  Dear Dave this is your final notice to collect...
    20     ham                             At 7 we will go ok na.
    21     ham          Are you willing to go for aptitude class.
    22     ham                          Are you comingdown later?
    23     ham   You are a very very very very bad girl. Or lady.
    24     ham        Yo! Howz u? girls never rang after india. L
    25     ham                                          Ok can...
    26     ham  A bit of Ur smile is my hppnss, a drop of Ur t...
    27     ham          Guessin you ain't gonna be here before 9?
    28    spam  URGENT! You have won a 1 week FREE membership ...
    29     ham  THING R GOOD THANX GOT EXAMS IN MARCH IVE DONE...
    ...    ...                                                ...
    1084   ham  Can not use foreign stamps in this country. Go...
    1085   ham                    S s..first time..dhoni rocks...
    1086  spam  Urgent! Please call 09061743811 from landline....
    1087   ham  I got like $ &lt;#&gt; , I can get some more l...
    1088   ham  Wow ... I love you sooo much, you know ? I can...
    1089   ham                        Dont gimme that lip caveboy
    1090   ham           Die... Now i have e toot fringe again...
    1091   ham  I know I'm lacking on most of this particular ...
    1092   ham                   Thanx 4 e brownie it's v nice...
    1093   ham                         Prepare to be pleasured :)
    1094  spam  Text BANNEDUK to 89555 to see! cost 150p texto...
    1095   ham  Wen ur lovable bcums angry wid u, dnt take it ...
    1096   ham  Bognor it is! Should be splendid at this time ...
    1097   ham  I'm doing da intro covers energy trends n pros...
    1098   ham  I've told you everything will stop. Just dont ...
    1099   ham  Do u konw waht is rael FRIENDSHIP Im gving yuo...
    1100   ham           As in i want custom officer discount oh.
    1101   ham                               I know she called me
    1102   ham  K.. I yan jiu liao... Sat we can go 4 bugis vi...
    1103   ham  Tell your friends what you plan to do on Valen...
    1104   ham  Double eviction this week - Spiral and Michael...
    1105   ham         I know you are. Can you pls open the back?
    1106   ham  Am on a train back from northampton so i'm afr...
    1107   ham                   K...k...yesterday i was in cbe .
    1108   ham  ARR birthday today:) i wish him to get more os...
    1109   ham  We're all getting worried over here, derek and...
    1110   ham  Oh oh... Den muz change plan liao... Go back h...
    1111   ham  CERI U REBEL! SWEET DREAMZ ME LITTLE BUDDY!! C...
    1112  spam  Text & meet someone sexy today. U can find a d...
    1113   ham                            K k:) sms chat with me.
    
    [1114 rows x 2 columns]



```python
#Find the percentage of spam and ham in both the training and the test set. 

training_set['Label'].value_counts(normalize=True)
```




    ham     0.86541
    spam    0.13459
    Name: Label, dtype: float64




```python
test_set['Label'].value_counts(normalize=True)
```




    ham     0.868043
    spam    0.131957
    Name: Label, dtype: float64



## Step2: S - Scrubbing / Cleaning our data
the SMS column is replaced by a series of new columns, where each column represents a unique word from the vocabulary.

### Letter Case and Punctuation


```python

# remove all the punctuation and bringing every letter
training_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Yep, by the pretty sculpture</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Yes, princess. Are you going to make me moan?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>Welp apparently he retired</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>Havent.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>I forgot 2 ask ü all smth.. There's a card on ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# use re.sub('\W', ' ',) for split the word
training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ')
training_set['SMS'] = training_set['SMS'].str.lower()
training_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>yep  by the pretty sculpture</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>yes  princess  are you going to make me moan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>welp apparently he retired</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>havent</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>i forgot 2 ask ü all smth   there s a card on ...</td>
    </tr>
  </tbody>
</table>
</div>



### Creating the Vocabulary

1. Let's now move to creating the vocabulary, which in this context means a list with all the unique words in our training set.


```python
vocabulary = []

training_set['SMS'] = training_set['SMS'].str.split()


for sms in training_set['SMS']:
    for word in sms:
        vocabulary.append(word)
        
vocabulary = list(set(vocabulary))
```


```python
len(vocabulary)
```




    7783



### The Final Traning Set
 we are going to use the vocabulary we just created make the dada transformation we want


```python
# get the word_counts_per_sms dictionary.
word_counts_per_sms= {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index , sms in enumerate(training_set['SMS']):
    for word in sms:
         word_counts_per_sms[word][index] += 1
```


```python
#Transform word_counts_per_sms into a DataFrame using pd.DataFrame().
word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>00</th>
      <th>000</th>
      <th>000pes</th>
      <th>008704050406</th>
      <th>0089</th>
      <th>01223585334</th>
      <th>02</th>
      <th>0207</th>
      <th>02072069400</th>
      <th>...</th>
      <th>zindgi</th>
      <th>zoe</th>
      <th>zogtorius</th>
      <th>zouk</th>
      <th>zyada</th>
      <th>é</th>
      <th>ú1</th>
      <th>ü</th>
      <th>〨ud</th>
      <th>鈥</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7783 columns</p>
</div>




```python
training_set_clean = pd.concat([training_set, word_counts], axis=1)

training_set_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
      <th>0</th>
      <th>00</th>
      <th>000</th>
      <th>000pes</th>
      <th>008704050406</th>
      <th>0089</th>
      <th>01223585334</th>
      <th>02</th>
      <th>...</th>
      <th>zindgi</th>
      <th>zoe</th>
      <th>zogtorius</th>
      <th>zouk</th>
      <th>zyada</th>
      <th>é</th>
      <th>ú1</th>
      <th>ü</th>
      <th>〨ud</th>
      <th>鈥</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>[yep, by, the, pretty, sculpture]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>[yes, princess, are, you, going, to, make, me,...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>[welp, apparently, he, retired]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>[havent]</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>[i, forgot, 2, ask, ü, all, smth, there, s, a,...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7785 columns</p>
</div>



# M-Model 
1. Calculating Constants
2. Calculating Parameters
3. Classifying A New Message


## Calculating Constants
After we have done with cleaning the training set, and we can begin creating the spam filter. The Naive Bayes algorithms will need to answer these two probability questions to be able to classify new messages:

$$P(Spam|w_1,\cdots,w_n)\propto P(Spam) \Pi_{i=1}^{n}P(w_i|Spam)$$
$$P(Ham|w_1,\cdots,w_n)\propto P(Ham) \Pi_{i=1}^{n}P(w_i|Ham)$$

Also, to calculate $P(w_i|Spam)$ and $P(w_i|Spam)$ inside the formulas above, we will need to use these equation:
$$P(w_i|Spam)=\frac{N_{w_i|Spam}+\alpha}{N_{Spam}+\alpha N_{Vocabulary}}$$
$$P(w_i|Ham)=\frac{N_{w_i|Ham}+\alpha}{N_{ham}+\alpha N_{Vocabulary}}$$

Last, we also need:

1. $P(Spam)$ and $P(Ham)$
2. $N_{spam}$, $N_{ham}$, $N_{vocabulary}$

Here we consider Laplace smoothing and set $\alpha=1$


```python
# Isolating spam and ham messages first
spam_messages = training_set_clean[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['Label'] == 'ham']
```


```python

# P(Spam) and P(Ham)
p_spam = len(spam_messages) / len(training_set_clean)
p_ham = len(ham_messages) / len(training_set_clean)

# N_Spam
n_words_per_spam_message = spam_messages['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham
n_words_per_ham_message = ham_messages['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1
```

## Calculating Parameters
1. now that we ahve the constant terms calculated above, we cvan move on with calculateing the parameters $P(w_i|Spam)$ and $P(w_i|Ham)$. Each parameter will thus be a conditional probability value associated with each word in the vocabulary.
2. The parameters are claculated using the formulas:
$$P(w_i|Spam)= \frac{N_{(w_i|Spam)}+\alpha}{N_{Spam}+\alpha *N_{vocabulary}}$$
$$P(w_i|Ham)= \frac{N_{(w_i|Ham)}+\alpha}{N_{Ham}+\alpha *N_{vocabulary}}$$



```python
# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
    n_word_given_spam = spam_messages[word].sum()   # spam_messages already defined in a cell above
    
    p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
    
    parameters_spam[word] = p_word_given_spam
    
    n_word_given_ham = ham_messages[word].sum()   # ham_messages already defined in a cell above
    
    p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
    
    parameters_ham[word] = p_word_given_ham
```

## Classifying A New Message
1. we have calculated all parameters, now it is time for the spam filter. 
2. The spam filter can be represented as:
   1. Takes in as input a new message $(w_1, w_2, ..., w_n)$.
   2. Calculates $P(Spam| w_1,w_2,...,w_n)$ and $P(Ham| w_1,w_2,...,w_n)$
   3. Compares the values of $P(Spam| w_1,w_2,...,w_n)$ and $P(Ham| w_1,w_2,...,w_n)$, and:
      1. If $P(Spam| w_1,w_2,...,w_n)$>$P(Ham| w_1,w_2,...,w_n)$, then the message is classified as spam.
      2. If $P(Spam| w_1,w_2,...,w_n)$<$P(Ham| w_1,w_2,...,w_n)$, then the message is classified as ham.
      3. If $P(Spam| w_1,w_2,...,w_n)$=$P(Ham| w_1,w_2,...,w_n)$, then the algorithm may request human help.
      


```python
import re

def classify(message):
    '''
    message: a string
    '''
    
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
            
    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)
    
    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal proabilities, have a human classify this!')
```

Now that we have a function that returns labels instead of printing them, we can use it to create a new column in our test set.


```python
classify('WINNER!! This is the secret code to unlock the money: C3421.')
```

    P(Spam|message): 1.3481290211300841e-25
    P(Ham|message): 1.9368049028589875e-27
    Label: Spam



```python
classify("Sounds good, Tom, then see u there")
```

    P(Spam|message): 2.4372375665888117e-25
    P(Ham|message): 3.687530435009238e-21
    Label: Ham


# N- Interpreting the data


```python
def classify_test_set(message):
    '''
    message: a string
    '''
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
            
        if p_ham_given_message > p_spam_given_message:
            return 'ham'
        elif p_spam_given_message > p_ham_given_message:
            return 'spam'
        else:
            return 'needs human classification'
        

```

Now that we have a function that returns labels instead of printing them, we can use it to create a new column in our test set


```python
test_set['predicted'] = test_set['SMS'].apply(classify_test_set)
test_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
      <th>predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Later i guess. I needa do mcat study too.</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>But i haf enuff space got like 4 mb...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Had your mobile 10 mths? Update to latest Oran...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>All sounds good. Fingers . Makes it difficult ...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>All done, all handed in. Don't know if mega sh...</td>
      <td>ham</td>
    </tr>
  </tbody>
</table>
</div>



now we will write a function to measure the accuracy of our spam fileter to find out how well our spam filter does.

# N - Interpreting our data


```python
correct = 0
total = test_set.shape[0]

for row in test_set.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct +=1
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)
```

    Correct: 1010
    Incorrect: 104
    Accuracy: 0.9066427289048474


The accuracy tells us, if there are 1,1114 messages that it has not seen in training,  we claim around 10000 message correctly.

# Next Steps

1. In conclusion, we success to build a spam filter for SMS messages using the multinomial Naive Bayes algorithm. The filter had an accuracy of our test set. 
2. Next steps include:
    1. analyze the incorrectly messages and try to figure out why the algorithm classified them incrrectly
    2. Make the filtering process more complex by making the algorithm sensitive to letter case



```python

```
