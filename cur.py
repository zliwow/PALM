import os
import json
import sys
from pathlib import Path
import pandas as pd
import shutil

# Set environment variables for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class PALMGenerator:
    def __init__(self, output_dir):
        self.base_dir = Path(os.getcwd())
        self.config_path = self.base_dir / "Code/config/common/seq2seq_generate.json"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_config(self, heavy_chain, light_chain, cdrh3_begin, cdrh3_end, antigen=None):
        """Create configuration for PALM-H3"""
        config = {
            "name": "CoV_AbDab-Seq2seq-Evaluate-Common",
            "n_gpu": 0,  # Use CPU/MPS
            "data_loader": {
                "type": "AntibodyAntigenSeq2SeqDataset",
                "args": {
                    "seed": 0,
                    "data_dir": "../ProcessedData/CoV_AbDab/CoV-AbDab_only_sars2_filter_only_1_cdr3.csv",
                    "antibody_vocab_dir": "../ProcessedData/vocab/antibody-2-3.csv",
                    "antigen_vocab_dir": "../ProcessedData/vocab/beta-2-3.csv",
                    "antibody_tokenizer_dir": "../Result_covid_heavy/checkpoints/BERT-Pretrain-common-MAA-NGPUs/pretrained",
                    "antigen_tokenizer_dir": "../Result_covid_heavy/checkpoints/BERT-Pretrain-common-MAA-NGPUs/pretrained",
                    "antibody_tokenizer_name": "common",
                    "antigen_tokenizer_name": "common",
                    "antibody_token_length_list": "2,3",
                    "antigen_token_length_list": "2,3",
                    "antibody_seq_name": "cdr3",
                    "antigen_seq_name": "antigen",
                    "antibody_max_len": 16,
                    "antigen_max_len": 300,
                    "encoder_input": "antigen",
                    "valid_split": 0.1,
                    "shuffle": True
                }
            },
            "origin_seq": heavy_chain,
            "origin_light": light_chain,
            "cdrh3_begin": cdrh3_begin,
            "cdrh3_end": cdrh3_end,
            "use_antigen": antigen if antigen else "RVQPTESIVRFPNITNLCPFHEVFNATTFASVYAWNRKRISNCVADYSVIYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKPSGNYNYLYRLFRKSKLKPFERDISTEIYQAGNKPCNGVAGSNCYSPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF",
            "TransformerVariant": "Antigen-Antibody",
            "resume": "../Result_seq2seq/checkpoints/ABAG-Finetuning-Seq2seq-Common/pretrained",
            "trainer": {
                "save_dir": "../Result_seq2seq_gen/",
                "verbosity": 2
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Created config file at: {self.config_path}")
        return self.config_path

    def generate(self):
        """Run the generation command"""
        if not self.config_path.exists():
            raise FileNotFoundError("Config file not found. Run create_config first.")
        
        original_dir = os.getcwd()
        os.chdir(self.base_dir / "Code")
        
        cmd = f"python generate_antibody.py --config ./config/common/seq2seq_generate.json"
        print(f"Running command: {cmd}")
        return_code = os.system(cmd)
        
        if return_code == 0:
            print("Generation completed successfully!")
            result_path = self.base_dir / "Result_seq2seq_gen/datasplit/CoV_AbDab-Seq2seq-Evaluate-Common"
            
            try:
                result_files = list(result_path.glob('*/result.csv'))
                if result_files:
                    newest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                    results_df = pd.read_csv(newest_file)
                    
                    # Take only first 10 sequences
                    results_df = results_df.head(10)
                    
                    # Save to output directory
                    output_file = self.output_dir / 'generated_sequences.csv'
                    results_df.to_csv(output_file, index=False)
                    
                    print("\nGeneration Results:")
                    print("-" * 50)
                    print(f"Total sequences generated: {len(results_df)}")
                    print("\nGenerated sequences:")
                    print(results_df[['Generated_CDR_H3', 'Heavy_Chain']].to_string())
                    print(f"\nResults saved to: {output_file}")
            except Exception as e:
                print(f"Could not read results: {e}")
                
        else:
            print("Generation failed with error code:", return_code)
        
        os.chdir(original_dir)

if __name__ == "__main__":
    # Your sequences
    heavy_chain = "QVQLVQSGAEVKKPGASVKVSCKVSGYTFTSHAIHWVRQAPGKGLEWMGGISPYRGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDAGVPLDYWGQGTLVTVSS"
    light_chain = "DIQMTQSPSSVSASVGDRVTITCRASQSIGTHLAWYQQKPGKAPKLLIYGASNLESGVPSRFSGSGSGTDFTLTISSLQPEDFANYYCQQYKAYPLTFGGGTKVEIK"
    cdrh3_begin = 98
    cdrh3_end = 107
    output_dir = "/Users/vka0479/Library/CloudStorage/OneDrive-Takeda/Desktop/Work Dir/PALM/results"

    generator = PALMGenerator(output_dir)
    generator.create_config(
        heavy_chain=heavy_chain,
        light_chain=light_chain,
        cdrh3_begin=cdrh3_begin,
        cdrh3_end=cdrh3_end
    )
    generator.generate()