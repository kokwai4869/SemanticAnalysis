# Import all the necessary libraries

import os
import json
import numpy as np
import yat
from keras.models import Sequential
from keras.layers import Embedding, Dense, Bidirectional, LSTM, \
    Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

#Get the dataset
cwd = os.getcwd()
with open("yahoo-movie-reviews.json", encoding="utf-8") as f:
    j = json.load(f)

data = []
for x in j:
    if x["movieName"] == "おくりびと":
        data.append(x)

reviews = {}
for x in data:
    reviews[x["text"]] = x["rating"]

#Separate the data based on 80%

num_data = int(80/100 * len(data))
    
x_train = list(reviews.keys())[:num_data]
y_train = list(reviews.values())[:num_data]
x_test = list(reviews.keys())[num_data:]
y_test = list(reviews.values())[num_data:]


#Create vocabulary for the total data

x_data = reviews.keys()
y_data = reviews.values()

tokenizer = yat.Tokenizer()
tokenizer.fit_on_texts(x_data)
vocabulary = {k[0]:v for k, v in tokenizer.token2id.items()}
reverse_vocabulary = dict(zip(vocabulary.values(), vocabulary.keys()))

# tokenize the training data

x_tokens_train = tokenizer.texts_to_sequences(x_train)
x_tokens_test = tokenizer.texts_to_sequences(x_test)
num_tokens = [len(token) for token in x_tokens_train + x_tokens_test]
num_tokens = np.array(num_tokens)
max_num = np.mean(num_tokens) + 2*np.std(num_tokens)
max_num = int(max_num)
x_train_padded = pad_sequences(x_tokens_train, maxlen=max_num, padding="pre",
                                truncating="pre")
x_test_padded = pad_sequences(x_tokens_test, maxlen=max_num, padding="pre",
                                truncating="pre")

# Create the model
embedding_size = 8
model = Sequential()
model.add(Embedding(input_dim=20000, output_dim=16, input_length=max_num,
                    name="embedding_layer"))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(1,activation="sigmoid"))
optimizer = Adam(lr=1e-13)

#Compile and train the model
model.compile(optimizer=optimizer, loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_train_padded, y_train, batch_size=64, epochs=1000,
          validation_data=[x_test_padded, y_test])

#Test the model

text2 = "人は誰でもいつかは死に直面する。\n\n葬式・・それは人生最後の主役の時。\n\nそれを人生で１番、美しく飾る納棺師（おくりびと）。\n\n今まで幸いにも葬式に縁のなかった自分はこの映画で、\n初めて納棺師の存在を知った。\n\n「今までで１番綺麗だった。」\n家族からの言葉。\n\nこんなに素晴らしい職業なのに\n世間からは偏見を受けている。\n\n周りからは反対され、妻からも「触らないで！！」\nと言われてしまう。\n結婚式の「美」と何ら変わらないはずなのに・・(T-T)。\n\n このテーマを15年も温め続け、\nこのような素晴らしい作品として「映画」という形にした本木雅弘。\n演技にもその執念が感じられました。\nこの傑作は彼の執念なしには\n生まれる事はなかったであろう。\n\nこの『おくりびと』を観て感じた事は、\n生や死とか、「納棺師」の美ではなく、\nどんな職業であれ自分の職業に「誇り」を持つ事、\n「誇り」を持てる職業を見つけ精進するという事である。\n\nそれが本当の「美」を生み出すのではないか？\n\n10年後もう一度観たい！\nアカデミー賞おめでとうございます♪"
text3 = "人生の最後を他人に託す”考えてみれば凄い仕事だ。当事者と共に多くの人生を生きてきた人達の前で行われる旅立ちの儀。精神を研ぎ澄まし、細心の注意を払い行われる、精神的にもキツイ仕事だろう。その仕事に関わる事と成った主人公小林大悟が、納棺師という仕事を通し人生を見詰め直す物語になっている。"
text4 = "タイトルからして\u3000泣いてしまうだろう…と覚悟して見始めたんだけど、\nプッと笑ったり、思わず\u3000アハハ＊と笑ってしまってた！\n\nそして\u3000やっぱり\u3000涙が溢れて…溢れて…\n\nいろんな人生があり\u3000いろんな死がある。\nそして、\nそれを見守る家族や関わりのある人たちがいる…\n\nそれを本当にあたたかく、せつなく、やさしく見せてくれる作品でした。\n\n本木雅弘の凛とした姿が素敵だったし、\n山崎努\u3000吉行和子\u3000笹野高史\n何か\u3000とっても自然で\u3000心揺さぶられました。\n\n父が亡くなって３年になるのですが、\n父の時の納棺師の方はまだ若い女性の方でした。\n本当に、丁寧にきれいにしていただきました。\n一番悲しい時に\n遺体になってしまった父を\u3000大切に扱ってくださったことは、\n残された私たちには\u3000何よりもありがたく\n感謝の気持ちでいっぱいだったことを思い出します。\n\n\n映画に映し出される\u3000手\n固くこわばった手が、\n亡くなったという事実とその人の人生を表しているような気がして\nせつなく、悲しかった…\nそして、\nその手を包み込む本木雅弘のあたたかい手…\n本当にやさしく胸がいっぱいになりました。\n\nいろんな場面で父を思い、父と一緒に見たような気がします。\nいい映画でした…\n\nありがとう…"
text6 = "良いですね"
text7 = "やばい"
texts = [text2, text3, text4,text6, text7]

texts = tokenizer.texts_to_sequences(texts)
texts = pad_sequences(texts, maxlen=max_num, padding="pre", truncating="pre")
predicted = model.predict(texts)
print(predicted)
sentiment = ["good" if x > 3 else "bad" for x in predicted]
for i in range(len(sentiment)):
    print("text {}: {}".format(i+1, sentiment[i]))





