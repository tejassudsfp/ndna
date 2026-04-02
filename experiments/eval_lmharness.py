"""Evaluate genome GPT-2 on standard benchmarks using lm-evaluation-harness."""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import tiktoken
from genome.model import GrownGPT2, Genome
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import lm_eval


@register_model('grown_gpt2')
class GrownGPT2LM(LM):
    def __init__(self, checkpoint_path, device='cuda', batch_size=16, **kwargs):
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = int(batch_size)

        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=self._device)
        cfg = ckpt['config']
        model_kwargs = dict(vocab_size=cfg['vocab_size'], hidden=cfg['hidden'],
                           ff_dim=cfg['ff_dim'], n_layers=cfg['n_layers'],
                           n_heads=cfg['n_heads'], max_len=cfg['block_size'])

        n_bands = cfg['n_layers'] + 2
        genome = Genome(n_types=cfg['n_types'], type_dim=cfg['type_dim'], n_bands=n_bands)
        self.model = GrownGPT2(genome, **model_kwargs)

        state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state'].items()}
        # For pruned model, genome keys may be missing - load with strict=False
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self._device)
        self.model.eval()
        # Set hard masks so genome masks are binary
        self.model.hard_masks = True

        self.tokenizer = tiktoken.get_encoding('gpt2')
        self._eot_token_id = self.tokenizer.eot_token
        self._max_length = cfg['block_size']
        print(f'  Loaded model: {sum(p.numel() for p in self.model.parameters()):,} params')

    @property
    def eot_token_id(self): return self._eot_token_id
    @property
    def max_length(self): return self._max_length
    @property
    def max_gen_toks(self): return 256
    @property
    def batch_size(self): return self._batch_size
    @property
    def device(self): return self._device

    def tok_encode(self, string, **kwargs): return self.tokenizer.encode(string)
    def tok_decode(self, tokens, **kwargs): return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        results = []
        for req in requests:
            ctx, cont = req.args
            ctx_tokens = self.tokenizer.encode(ctx)
            cont_tokens = self.tokenizer.encode(cont)
            all_tokens = ctx_tokens + cont_tokens
            if len(all_tokens) > self._max_length:
                all_tokens = all_tokens[-self._max_length:]
                ctx_len = len(all_tokens) - len(cont_tokens)
            else:
                ctx_len = len(ctx_tokens)

            x = torch.tensor([all_tokens[:-1]], dtype=torch.long, device=self._device)
            with torch.no_grad():
                logits, _ = self.model(x)
            log_probs = F.log_softmax(logits, dim=-1)
            cont_start = ctx_len - 1
            cont_ll = 0.0
            is_greedy = True
            for i in range(cont_start, len(all_tokens) - 1):
                tid = all_tokens[i + 1]
                cont_ll += log_probs[0, i, tid].item()
                if logits[0, i].argmax().item() != tid:
                    is_greedy = False
            results.append((cont_ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for req in requests:
            (string,) = req.args
            tokens = self.tokenizer.encode(string)
            total_ll = 0.0
            for i in range(0, len(tokens), self._max_length):
                chunk = tokens[i:i + self._max_length + 1]
                if len(chunk) < 2: continue
                x = torch.tensor([chunk[:-1]], dtype=torch.long, device=self._device)
                with torch.no_grad():
                    logits, _ = self.model(x)
                lp = F.log_softmax(logits, dim=-1)
                for j in range(len(chunk) - 1):
                    total_ll += lp[0, j, chunk[j + 1]].item()
            results.append((total_ll,))
        return results

    def generate_until(self, requests):
        raise NotImplementedError('Generation not needed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tasks', default='hellaswag,lambada_openai,winogrande,piqa,arc_easy')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    print('=' * 60)
    print('  GENOME GPT-2 BENCHMARK (lm-evaluation-harness)')
    print('=' * 60)

    results = lm_eval.simple_evaluate(
        model='grown_gpt2',
        model_args=f'checkpoint_path={args.checkpoint},device={args.device},batch_size={args.batch_size}',
        tasks=args.tasks.split(','),
        batch_size=None,
    )

    print('\n' + '=' * 60)
    print('  RESULTS')
    print('=' * 60)
    for task_name, task_results in results['results'].items():
        print(f'\n  {task_name}:')
        for metric, value in task_results.items():
            if metric != 'alias' and not metric.endswith('_stderr'):
                stderr_key = f'{metric}_stderr'
                stderr = task_results.get(stderr_key, None)
                if stderr is not None:
                    print(f'    {metric}: {value:.4f} +/- {stderr:.4f}')
                elif isinstance(value, float):
                    print(f'    {metric}: {value:.4f}')

    os.makedirs('results/gpt2_full', exist_ok=True)
    with open('results/gpt2_full/eval_genome_lmharness.json', 'w') as f:
        json.dump(results['results'], f, indent=2)
    print(f'\n  Saved to results/gpt2_full/eval_genome_lmharness.json')
