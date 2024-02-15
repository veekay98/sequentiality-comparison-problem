import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math
from nltk import tokenize
import numpy as np
import matplotlib.pyplot as plt


# Reading the csv file that contains all the stories and summaries
df = pd.read_csv("hippocorpus-u20220112/hippoCorpusV2.csv")

#-------------------------------------------------------------
df_imagined=df[df['memType']=='imagined'] # The dataframe containing all imagined stories
df_retold=df[df['memType']=='retold'] # The dataframe containing all retold stories
df_recalled=df[df['memType']=='recalled'] # The dataframe containing all recalled stories

#----------------------------------------------------------------

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

#--------------------------------------------------------------------

# Finding the recalled rows for passing to the model
mask = df['AssignmentId'].isin(df_retold['recAgnPairId']) # finding connection between recalled and retold
matching_rows = df[mask] # the rows of recalled that are connected to retold
temp=matching_rows # storing in temp
mask = temp['AssignmentId'].isin(df_imagined['recImgPairId']) # we find the rows from retold-recalled that are connected to imagined
matching_rows = temp[mask]
mask2= df['AssignmentId'].isin(temp['AssignmentId'])
matching_rows_recalled = df[mask2] # the recalled rows that are linked to imagined and retold

# Finding the retold rows for passing to the model
mask_retold = df_retold['recAgnPairId'].isin(temp['AssignmentId'])
matching_rows_retold=df_retold[mask_retold] # the retold rows that are linked to recalled

# Finding the imagined rows for passing to the model
mask_imagined = df_imagined['recImgPairId'].isin(temp['AssignmentId'])
matching_rows_imagined=df_imagined[mask_imagined] # the imagined rows that are linked to recalled

#-----------------------------------------------------------------

# Function to calculate the log likelihood by passing the target sentence and the prompt to GPT2
def calculate_log_likelihood(target_sentence, prompt):

        likelihood = 1.0
        log_likelihood = 0
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")

        prompt_tokens_length = len(prompt_tokens[0])
        target_sentence_tokens = tokenizer.encode(target_sentence, return_tensors="pt")

        combined_token = torch.cat((prompt_tokens, target_sentence_tokens), 1)

        if model == "gpt3":
            return self.askGPT(prompt + " " + target_sentence, prompt_tokens_length)

        output = model(combined_token)
        logits = output[0]
        with torch.no_grad():

            for i in range(prompt_tokens_length, len(combined_token[0])):
                next_token_prob = torch.softmax(logits[:, i - 1, :], dim=-1)
                # Get the index of the next token in the sequence
                next_token_index = combined_token[0][i]

                # Get the likelihood of the actual next token
                token_likelihood = next_token_prob[0][next_token_index].item()

                # Multiply the likelihood with the running total
                likelihood *= token_likelihood
                log_likelihood += math.log(token_likelihood, 2)


        return log_likelihood


#----------------------------------------------------------------------

# Function to calculate the sequentiality for imagined, retold and recalled stories for history size 1 to 9 and for full size history
def calculate_seq_per_history(dataset, history, full_size, story_type):

    sum_common=0 # Sum for the final sequentiality

    sum_topic=0 # Sum for the topic NLLs

    sum_context=0 # Sum for the topic+context NLLs

    counter=0 # Counts the number of rows encountered so far.
    # Kept as a seperate var from j in case we want to iterate through different range in the dataset

    # Iterates through all the rows
    for j in range(len(dataset)):
        index=j

        topic=dataset['summary'].iloc[index] # The topic in the study is the value of summary column
        story=dataset['story'].iloc[index] # The story value is that of the story column

        sentences = sent_tokenize(story) # This discovers the sentences in the story

        arr=[] # Array for storing sequentiality for each combination of sentences in the story

        arr_topics=[] # Array for storing NLL for just the topic

        arr_context=[] # Array for storing NLL for the topic+context

        # The sentences have to be at least of size 9 to find sequentiality for history size 9.
        # This check is done for all types of stories for consistency
        if (len(sentences)<=9):
            continue

        # history=history_img # Except for full size history, this value is same for
        if (full_size=="true"):  # For the case of full size history
            history=len(sentences)-1

        for i in range(len(sentences)-history):

            sentence_context=""

            for k in range(history):
                sentence_context+=" "
                sentence_context+=sentences[i+k]

            prompt1=topic
            prompt2=topic+sentence_context


            l_topic = calculate_log_likelihood(sentences[i+history], prompt1) # Returns the log likelihood for target sentence from just the topic as prompt
            l_topic_context = calculate_log_likelihood(sentences[i+history], prompt2) # Returns the log likelihood for target sentence from both the topic and context as prompt
            arr_topics.append((-1)/len(sentences[i+history])*(l_topic))
            arr_context.append((-1)/len(sentences[i+history])*(l_topic_context))
            arr.append(1/len(sentences[i+history])*(l_topic_context-l_topic))

        avg=np.mean(arr)
        avg_topics=np.mean(arr_topics)
        avg_context=np.mean(arr_context)


        sum_common+=avg
        sum_topic+=avg_topics
        sum_context+=avg_context

        counter+=1


    # print("FINAL COUNT IS",counter,"\n")
    print("FINAL SEQUENTIALITY FOR",story_type,"FOR HISTORY",history,"IS",sum_common/counter,"\n")

    print("FINAL TOPIC NLL FOR",story_type,"FOR HISTORY",history,"IS",sum_topic/counter,"\n")
    print("FINAL CONTEXT NLL FOR",story_type,"FOR HISTORY",history,"IS",sum_context/counter,"\n")

    return sum_topic/counter, sum_context/counter, sum_common/counter




final_seq_array=[] # Array to store the history sizes and the final sequentiality values for imagined, retold, recalled
final_topic_nll_array=[] # Array to store the history sizes and the final topic NLL values for imagined, retold, recalled
final_context_nll_array=[] # Array to store the history sizes and the final context NLL values for imagined, retold, recalled

for i in range(1,10):
    print("FOR HISTORY VALUE",i)
    imagined_topic_nll, imagined_context_nll, imagined_seq=calculate_seq_per_history(matching_rows_imagined, i, "false","IMAGINED")
    retold_topic_nll, retold_context_nll, retold_seq=calculate_seq_per_history(matching_rows_retold, i, "false","RETOLD")
    recalled_topic_nll, recalled_context_nll, recalled_seq=calculate_seq_per_history(matching_rows_recalled, i, "false","RECALLED")
    final_seq_array.append([i,imagined_seq,retold_seq,recalled_seq])
    final_topic_nll_array.append([i,imagined_topic_nll,retold_topic_nll,recalled_topic_nll])
    final_context_nll_array.append([i,imagined_context_nll,retold_context_nll,recalled_context_nll])
    print("STATUS OF FINAL SEQUENTIALITY ARRAY UPTO HISTORY",i,"IS",final_seq_array)
    print("STATUS OF FINAL TOPIC NLL ARRAY UPTO HISTORY",i,"IS",final_topic_nll_array)
    print("STATUS OF FINAL CONTEXT NLL ARRAY UPTO HISTORY",i,"IS",final_context_nll_array)

print("FOR HISTORY VALUE OF FULL SIZE")
imagined_seq=calculate_seq_per_history(matching_rows_imagined, 0,"true","IMAGINED")
retold_seq=calculate_seq_per_history(matching_rows_retold, 0,"true","RETOLD")
recalled_seq=calculate_seq_per_history(matching_rows_recalled, 0,"true","RECALLED")
final_seq_array.append([100,imagined_seq,retold_seq,recalled_seq]) # Randomly assigned 100 for full size
final_topic_nll_array.append([100,imagined_topic_nll,retold_topic_nll,recalled_topic_nll])
final_context_nll_array.append([100,imagined_context_nll,retold_context_nll,recalled_context_nll])

print("STATUS OF FINAL SEQUENTIALITY ARRAY IS",final_seq_array)
print("STATUS OF FINAL TOPIC NLL ARRAY IS",final_topic_nll_array)
print("STATUS OF FINAL CONTEXT NLL ARRAY IS",final_context_nll_array)


#--------------------------------------------------------------------------------
# Plotting the graph

x = ['Imagined', 'Retold', 'Recalled'] # The three story types
ys=[] # The sequentiality values for all combinations
for i in range(len(final_seq_array)):
    ys.append(final_seq_array[i][1:])
# This was the result array I got for final sequentiality
# The first 9 sub arrays correspond to history sizes 1 to 9 and the last is for full size history
# Each sub array in the array has first the imagined, then the retold and then the recalled sequentialities
# ys = [
    # [0.07705354056051734, 0.065814822542532, 0.05993178222286483],
    # [0.09798711006103662, 0.08439238717562615, 0.07506038293182737],
    # [0.10872582040863597, 0.09345286030496933, 0.0831979150917731],
    # [0.11615774946436304, 0.09868673790558996, 0.08779488922920833],
    # [0.12099208519168099, 0.10170760029771776, 0.08973877743528115],
    # [0.1239537604174997, 0.10350080909222624, 0.0912048564141837],
    # [0.12698923293286393, 0.10445921759552997, 0.09149822997958795],
    # [0.12868495166870556, 0.10500568307444352, 0.09196264812421348],
    # [0.13055817134824116, 0.104064222127566, 0.09107517073076461],
    # [0.13633413712861994, 0.10626571005878947, 0.0988105219918958],
# ]

# Making the plot
for i, y in enumerate(ys):
    if (i==9):
        plt.plot(x, y, marker='o', label=f'History: Full Size')
    else:
        plt.plot(x, y, marker='o', label=f'History {i + 1}')

plt.xlabel('Story Types')
plt.ylabel('Sequentiality')
plt.title('Sequentiality patterns for different history values and story types')
plt.legend(bbox_to_anchor=(1.0, 0.8))  # To show legend
plt.grid(True)  # To show grid
plt.tight_layout()  # Adjust spacing to prevent clipping
plt.show()
