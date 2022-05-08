import os
import tempfile
import logging
#from files2rouge import settings, utils
#from pyrouge import Rouge155


class Evaluator:
    def __init__(self):
        self.rouge_args = ['-c', 95, '-r', 1000, '-n', 2, '-a']
        self._deal_args()

    def _deal_args(self):
        self.rouge_args = " ".join([str(_) for _ in self.rouge_args])

    def _write_file(self, write_path, content):
        f = open(write_path, 'w')
        f.write("\n".join(content))
        f.close()

    def _split_rouge(self, input_sentence):
        res_list = input_sentence.split()
        res = {res_list[1].lower(): round(float(res_list[3]), 4)}
        return res

    def _calc_rouge(self, args):
        summ_path = args['summ_path']
        ref_path = args['ref_path']
        eos = args['eos']
        ignore_empty_reference = args['ignore_empty_reference']
        ignore_empty_summary = args['ignore_empty_summary']
        stemming = args['stemming']

        s = settings.Settings()
        s._load()
        with tempfile.TemporaryDirectory() as dirpath:
            sys_root, model_root = [os.path.join(dirpath, _) for _ in ["system", "model"]]
            utils.mkdirs([sys_root, model_root])
            ignored = utils.split_files(
                model_path=ref_path,
                system_path=summ_path,
                model_dir=model_root,
                system_dir=sys_root,
                eos=eos,
                ignore_empty_reference=ignore_empty_reference,
                ignore_empty_summary=ignore_empty_summary
            )
#            r = Rouge155(rouge_dir=os.path.dirname(s.data['ROUGE_path']), log_level=logging.ERROR, stemming=stemming)
#            r.system_dir = sys_root
#            r.model_dir = model_root
#            r.system_filename_pattern = r's.(\d+).txt'
#            r.model_filename_pattern = 'm.[A-Z].#ID#.txt'
#            data_arg = "-e %s" % s.data['ROUGE_data']
#            rouge_args_str = "%s %s" % (data_arg, self.rouge_args)
#            output = r.convert_and_evaluate(rouge_args=rouge_args_str)
#            res = self._get_info(output)
        return 0#res

    def _get_info(self, input_str):
        rouge_list = input_str.replace("---------------------------------------------",
                                       "").replace("\n\n", "\n").strip().split("\n")
        rouge_list = [rouge for rouge in rouge_list if "Average_F" in rouge]
        rouge_dict = {}
        for each in list(map(self._split_rouge, rouge_list)):
            rouge_dict.update(each)
        return rouge_dict

    def _calc_metrics_info(self, generated_corpus, reference_corpus):
        generated_corpus = [" ".join(generated_sentence) for generated_sentence in generated_corpus]
        reference_corpus = [" ".join(reference_sentence) for reference_sentence in reference_corpus]
        with tempfile.TemporaryDirectory() as path:
            generated_path = os.path.join(path, 'generated_corpus.txt')
            reference_path = os.path.join(path, 'reference_corpus.txt')
            self._write_file(generated_path, generated_corpus)
            self._write_file(reference_path, reference_corpus)

            calc_args = {
                'summ_path': generated_path,
                'ref_path': reference_path,
                'eos': '.',
                'ignore_empty_reference': False,
                'ignore_empty_summary': False,
                'stemming': True
            }
            #res = self._calc_rouge(calc_args)
        return 0#res

    def evaluate(self, generated_corpus, reference_corpus):
        return self._calc_metrics_info(generated_corpus=generated_corpus, reference_corpus=reference_corpus)
