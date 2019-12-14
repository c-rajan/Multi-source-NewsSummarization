#from extractive_summarization import extractive_summary
from rouge import Rouge
from os import listdir
from os.path import isfile, join

def eval(gt_text, arg_text, non_arg_text=None):
    if non_arg_text:
        length_arg = len(arg_text)
        length_no_arg = len(non_arg_text)
        fpr_values = []
        for arg_length in [220, 330, 440]:
            ratio_arg = arg_length/length_arg
            ratio_no_arg = (660-arg_length)/length_no_arg
            if ratio_arg > 0.3:
                summary_arg = extractive_summary(arg_text, min(ratio_arg,1))
            else:
                summary_arg = extractive_summary(arg_text, ratio_arg, 20, 200)
            summary_no_arg = extractive_summary(non_arg_text, ratio_no_arg, 20, 200)
            summary = summary_no_arg + summary_arg
            rouge = Rouge()
            score = rouge.get_scores(summary, gt_text)
            print(summary)
            print(score[0]['rouge-1'])
            sco = score[0]['rouge-1']
            fpr_values.append(sco['f'])
            fpr_values.append(sco['p'])
            fpr_values.append(sco['r'])
        return fpr_values
    else:
        summary = arg_text
        #length = len(arg_text)
        #ratio = 665/length
        #if ratio > 0.3:
        #    summary = extractive_summary(arg_text, min(ratio,1))
        #else:
        #    summary = extractive_summary(arg_text, ratio, 20, 200)
    print(summary)
    rouge = Rouge()
    score = rouge.get_scores(summary, gt_text)
    print(score)
    sco = score[0]['rouge-1']
    return sco['f'], sco['p'], sco['r']

def eval_average():
    path = '/mnt/disk_1/argument_classification/acl2019-BERT-argument-classification-and-clustering/argument-similarity/'
    path_all = path+'args_agglo/'
    files = [f for f in listdir(path_all) if isfile(join(path_all,f))]
    num_doc = len(files)
    rouge_f = {'normal':0, 'arg':0, 'mix1':0, 'mix2':0, 'mix3':0}
    rouge_r = {'normal':0, 'arg':0, 'mix1':0, 'mix2':0, 'mix3':0}
    rouge_p = {'normal':0, 'arg':0, 'mix1':0, 'mix2':0, 'mix3':0}
    for f in files:
        #initial = f[:-7]
        #path_all_file = path_all + f
        #path_arg_file = path + 'args/' + initial + 'args.txt'
        #path_no_arg_file = path + 'no_args/' + initial + 'no_args.txt'
        path_arg_agglo = path_all + f
        path_gt = '/home/asnani04/eval/models/2/single_summaries/'
        number = f[20:25]
        print(number)
        for gt_file in listdir(path_gt):
            if number in gt_file:
                path_gt_file = path_gt + gt_file
                break
        f_gt = open(path_gt_file, 'r')
        text_gt = f_gt.read()
        f_gt.close()

      #  f_all = open(path_all_file, 'r')
       # text_all = f_all.read()
        #f_all.close()

        f_arg = open(path_arg_agglo, 'r')
        text_arg = f_arg.read()
        f_arg.close()

       # f_no_arg = open(path_no_arg_file, 'r')
       # text_no_arg = f_no_arg.read()
       # f_no_arg.close()

       # score_f, score_p, score_r = eval(text_gt, text_all)
       # rouge_f['normal'] += score_f
       # rouge_r['normal'] += score_r
       # rouge_p['normal'] += score_p

        score_f, score_p, score_r = eval(text_gt, text_arg)
        rouge_f['arg'] += score_f
        rouge_r['arg'] += score_r
        rouge_p['arg'] += score_p

       # [score_f1, score_p1, score_r1, score_f2, score_p2, score_r2, score_f3, score_p3, score_r3] = eval(text_gt, text_arg, text_no_arg)
       ## rouge_f['mix'] += score_f
       # rouge_r['mix1'] += score_r1
       # rouge_r['mix2'] += score_r2
       # rouge_r['mix3'] += score_r3
       # rouge_p['mix1'] += score_p1
       # rouge_p['mix2'] += score_p2
       # rouge_p['mix3'] += score_p3

    rouge_r['normal'] = rouge_r['normal'] / num_doc
    rouge_r['mix1'] = rouge_r['mix1'] / num_doc
    rouge_r['mix2'] = rouge_r['mix2'] / num_doc
    rouge_r['mix3'] = rouge_r['mix3'] / num_doc
    rouge_r['arg'] = rouge_r['arg'] / num_doc
    rouge_p['normal'] = rouge_p['normal'] / num_doc
    rouge_p['mix1'] = rouge_p['mix1'] / num_doc
    rouge_p['mix2'] = rouge_p['mix2'] / num_doc
    rouge_p['mix3'] = rouge_p['mix3'] / num_doc
    rouge_p['arg'] = rouge_p['arg'] / num_doc
    print(rouge_f, rouge_r, rouge_p)

if __name__ == '__main__':
    eval_average()
