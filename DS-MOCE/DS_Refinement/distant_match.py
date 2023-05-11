# distant match for all the courses
# update: add the str_words and tags for each json file
import json
import os
from flashtext import KeywordProcessor
import pandas as pd
import jieba.posseg as pseg
import re



def get_all_text(dir_path):
    all_course_text = []
    for root, _, json_files in os.walk(dir_path):
        for json_file in json_files:
            with open(os.path.join(root, json_file)) as f:
                one_course_text = []
                this_course = json.load(f)
                for per_video in this_course['videos']:
                    one_course_text.append(per_video['text'])
                print("{} has {} videos".format(json_file, len(one_course_text)))
            all_course_text.append(one_course_text)
            print("{} json file done!".format(json_file))
    return all_course_text
                
def generate_whole_train(file_name,output_path,keyword_processor,fields_concepts_map):
    # with 512 cut off
    max_length = 500
    file_out = []
    # 3. match and generate tag_list:
    # debug: per course 1,000 sent.#
    cnt = 0
    for root, _, json_files in os.walk(file_name):
        for json_file in json_files:
            cnt += 1
            if cnt % 10 == 1:
                print("already process " ,cnt)
            # if cnt >= 15:
            #     continue
            with open(os.path.join(root,json_file),'r',encoding='utf-8') as f:
                this_course = json.load(f)
                for per_video in this_course['videos']:
                    count = max_length
                    words_list = []
                    tags_list = []
                    for per_sent in per_video['text']:
                        if count < len(per_sent):
                            dic = {"str_words": words_list+['。'], "tags": tags_list+[0]}
                            file_out.append(dic)
                            count = max_length
                            words_list = []
                            tags_list = []
                        else:
                            #w_list, tag_list = distant_match_sent(per_sent,keyword_processor,this_course['fields'],field_related,fields_concepts_map)
                            #w_list, tag_list = dictionary_match(per_sent,keyword_processor)
                            w_list,tag_list = pos_process_sent(per_sent,fields_concepts_map,this_course['fields'])
                            words_list = words_list + w_list +['，']
                            tags_list = tags_list + tag_list +[0]
                            count = count - len(per_sent)-1

                    if words_list:
                        dic = {"str_words": words_list+['。'], "tags": tags_list+[0]}
                        file_out.append(dic)

    # write:
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(file_out, f_out, ensure_ascii=False)

                
def generate_whole_test(file_name,fields_concepts_map,output_path):
    # with 512 cut off
    max_length = 500
    file_out = []
    # 3. match and generate tag_list:
    # debug: per course 1,000 sent.#
    cnt =0
    for root, _, json_files in os.walk(file_name):
        for json_file in json_files:
            cnt += 1
            if cnt % 10 == 1:
                print("already process " ,cnt)
            with open(os.path.join(root,json_file),'r',encoding='utf-8') as f:
                this_course = json.load(f)
                for per_video in this_course['videos']:
                    count = max_length
                    words_list = []
                    tags_list = []
                    for word,tag in zip(per_video['str_words'],per_video['tags']):
                        if count == 0:
                            dic = {"str_words": words_list, "tags": tags_list}
                            file_out.append(dic)
                            count = max_length
                            words_list = []
                            tags_list =[]
                        else:
                            words_list.append(word)
                            tags_list.append(tag)
                            count -= 1
                    
                    if words_list:
                        dic = {"str_words": words_list, "tags": tags_list}
                        file_out.append(dic)

    # write:
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(file_out, f_out, ensure_ascii=False)


def distant_match_sent(sent,keyword_processor,course_fields_list,field_related,fields_concepts_map=None):
    str_words = list(sent)
    tag_list = [0]*len(sent)

    concepts = keyword_processor.extract_keywords(sent, span_info=True)
    for word, s, e in concepts:
        if field_related:
            if list(set(fields_concepts_map[word])&set(course_fields_list)):
                # not empty:
                tag_list[s]=1
                for i in range(s+1,e):
                    tag_list[i]=2
        else:
            tag_list[s]=1
            for i in range(s+1,e):
                tag_list[i]=2
    
    assert(len(tag_list)==len(str_words))

    return str_words,tag_list


def dictionary_match(sent,keyword_processor):
    str_words = list(sent)
    tag_list = [0] * len(sent)

    concepts = keyword_processor.extract_keywords(sent, span_info=True)
    for word, s, e in concepts:
        tag_list[s] = 1
        for i in range(s + 1, e):
            tag_list[i] = 2

    assert (len(tag_list) == len(str_words))

    return str_words, tag_list

def distant_match(str_words,keyword_processor,course_fields_list,field_related,fields_concepts_map=None):
    # return tags[list]
    whole_text = ''.join(str_words)
    # BUG : why len(whole_text)!=len(str_words) 单字有:英文！

    tag_list = [0]*len(str_words)

    concepts = keyword_processor.extract_keywords(whole_text, span_info=True)
    for word, s, e in concepts:
        if field_related:
            if list(set(fields_concepts_map[word])&set(course_fields_list)):
                # not empty:
                tag_list[s]=1
                for i in range(s+1,e):
                    tag_list[i]=2
        else:
            tag_list[s]=1
            for i in range(s+1,e):
                tag_list[i]=2
    
    assert(len(tag_list)==len(str_words))

    return tag_list


def generate_for_each_json_file(tmp_dir,keyword_processor,fields_concepts_map=None):
    out_dir = '../tmp_train/'
    field_related = True
    for root, _, json_files in os.walk(tmp_dir):
        for json_file in json_files:
            with open(os.path.join(root,json_file),'r',encoding='utf-8') as f:
                this_course = json.load(f)
                for per_video in this_course['videos']:
                    tags = distant_match(per_video['str_words'],keyword_processor,this_course['fields'],field_related,fields_concepts_map) 
                    # update only tags because we have copied from ground_truth to distant_match,so no need to update text
                    per_video['tags'] = tags

                # write back:
                with open(os.path.join(out_dir,json_file),'w',encoding='utf-8') as f_out:
                    json.dump(this_course,f_out,ensure_ascii=False)

def generate_fields_concepts_file(file_name = None):
    # choose 1. construct dict from file: all_fields_concepts: including 20' classes 
    # with fields
    fields_concepts_map = {}

    keyword_processor = KeywordProcessor()
    # read file
    data = pd.read_csv(file_name) # make sure csv file has the header
    X = data.iloc[:,0].values.tolist()
    Y = data.iloc[:,1:].values.tolist()

    for index in range(len(X)):
        keyword_processor.add_keyword(X[index])
        fields_concepts_map[X[index]]=Y[index]

    return keyword_processor,fields_concepts_map

    # choose 2. construct dict from embeddings, get top1 field
    

def generate_only_concepts(file_name):
    # file_name : all_fields_concepts not original concept.json

    keyword_processor = KeywordProcessor()
    # read file
    data = pd.read_csv(file_name) # make sure csv file has the header
    X = data.iloc[:,0].values.tolist()
    # Y = data.iloc[:,1:].values.tolist()
    for per_concept in X:
        keyword_processor.add_keyword(per_concept)
    return keyword_processor

def generate_from_embedding(file_name):
    # we only take Top1?
    TopK = 2
    
    fields_concepts_map = {}
    keyword_processor = KeywordProcessor()

    # define:
    all_field = [
        '心理学', '教育学', '语言学',
        '世界历史', '数学', '物理学', '化学','机械工程',
        '材料科学技术', '电气工程',  '建筑学', 
        '航空科学技术', '医学', '管理科学技术', '计算机科学技术','农学',
        '力学','船舶工程','航天科学技术','免疫学',
    ]
    # read file
    data = pd.read_csv(file_name) # make sure csv file has the header
    X = data.iloc[:,0].values.tolist()
    Y = data.iloc[:,1:].values.tolist()

    for index in range(len(X)):
        keyword_processor.add_keyword(X[index])
        res = list(sorted(zip(all_field,Y[index]),key= lambda x:-x[1]))
        related_fields = []
        for i in range(TopK):
            related_fields.append(res[i][0])
        fields_concepts_map[X[index]]=related_fields
    
    return keyword_processor,fields_concepts_map

def is_noun(flag):
    if re.match(r'^(@(([av]?n[rstz]?)|l|a|v))*(@(([av]?n[rstz]?)|l))$', flag) is not None:
        return True
    else:
        return False

def pos_process_sent(sent,fields_concepts_map,course_fields_list):
    str_words = list(sent)
    tags_list = [0]*len(sent)

    tmp = pseg.cut(sent)
    seg = [(t.word, t.flag) for t in tmp]
    n = len(seg)
    tag_index = 0
    for i in range(n):
        phrase, flag = seg[i][0], '@'+seg[i][1]
        for j in range(i+1, min(n+1, i+7)):
            if phrase in fields_concepts_map.keys():
                if list(set(fields_concepts_map[phrase])&set(course_fields_list)) and is_noun(flag):
                    tags_list[tag_index]=1
                    for index in range(1,len(phrase)):
                        tags_list[tag_index+index]=2
            if j < n:
                phrase += seg[j][0]
                flag += '@'+seg[j][1]
        
        tag_index += len(seg[i][0])
    
    return str_words,tags_list

def pos_process(words_list,fields_concepts_map,course_fields_list):
    whole_text = ''.join(words_list)
    tags_list = [0]*len(words_list)

    tmp = pseg.cut(whole_text)
    seg = [(t.word, t.flag) for t in tmp]
    n = len(seg)
    tag_index = 0
    for i in range(n):
        phrase, flag = seg[i][0], '@'+seg[i][1]
        for j in range(i+1, min(n+1, i+7)):
            if phrase in fields_concepts_map.keys():
                if list(set(fields_concepts_map[phrase])&set(course_fields_list)) and is_noun(flag):
                    tags_list[tag_index]=1
                    for index in range(1,len(phrase)):
                        tags_list[tag_index+index]=2
            if j < n:
                phrase += seg[j][0]
                flag += '@'+seg[j][1]
        
        tag_index += len(seg[i][0])
    
    return tags_list




    

def handle_pos(input_dir,output_dir,fields_concepts_map):
    for root, _, json_files in os.walk(input_dir):
        for json_file in json_files:
            with open(os.path.join(root,json_file),'r',encoding='utf-8') as f:
                this_course = json.load(f)
                for per_video in this_course['videos']:
                    tags = pos_process(per_video['str_words'],fields_concepts_map,this_course['fields']) 
                    # update  tags 
                    per_video['tags'] = tags
                # write back:
                with open(os.path.join(output_dir,json_file),'w',encoding='utf-8') as f_out:
                    json.dump(this_course,f_out,ensure_ascii=False) 


if __name__ == '__main__':
    #keyword_processor = generate_only_concepts('all_fields_concepts.csv')
    keyword_processor,fields_concepts_map = generate_from_embedding('../Dictionary_Empowerment/data/output_embeddings.csv')
    generate_whole_train('./tmp_train/','output_train.json',keyword_processor,fields_concepts_map) 
    # keyword_processor,fields_concepts_map = generate_fields_concepts_file('all_fields_concepts.csv')
    
    # # 2. get text:
    # json_list = ['674920.json','676700.json','677247.json','681743.json','682303.json','682427.json',
    # '682716.json','696687.json','696826.json','696897.json','696927.json']
    # # json_list = ['674920.json']
    # train_path = '../tmp/'
    # all_course_text = get_all_text(train_path)

    #generate_whole_train('../tmp_train/','train20220702.json',keyword_processor,None)

    # generate_whole_test('../ground_truth/',fields_concepts_map,'test.json')

    #tmp_dir = '../tmp_train/'
    #generate_for_each_json_file(tmp_dir,keyword_processor,fields_concepts_map)

    # file_path = '../tmp_train/'
    # des_path = '../tmp_train/'
    # handle_pos(file_path,des_path,fields_concepts_map)


                




