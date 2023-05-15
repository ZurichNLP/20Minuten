#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers import MT5ForConditionalGeneration, T5Tokenizer

# see for improvinf eff: https://discuss.huggingface.co/t/using-trainer-at-inference-time/9378/7

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name_or_path', required=True, type=str)
    ap.add_argument('--tokenizer_name', required=False, default=None, type=str, help='if different from model name')
    ap.add_argument('--config_name', required=False, default=None, type=str, help='if different from model name')
    ap.add_argument('--max_output_length', required=False, default=64, type=int, help='')
    ap.add_argument('--prefix', required=False, default='summarize: ', type=str, help='')
    
    # generateion args
    ap.add_argument('--beam_size', required=False, default=4, type=int, help='')
    ap.add_argument('--num_return_sequences', required=False, default=1, type=int, help='')
    ap.add_argument('--do_sample', required=False, default=False, type=bool, help='')
    ap.add_argument('--top_k', required=False, default=50, type=int, help='')
    ap.add_argument('--top_p', required=False, default=1.0, type=float, help='')
    ap.add_argument('--temperature', required=False, default=1.0, type=float, help='')
    ap.add_argument('--length_penalty', required=False, default=1.0, type=float, help='')
    ap.add_argument('--repetition_penalty', required=False, default=1.0, type=float, help='')
    ap.add_argument('--diversity_penalty', required=False, default=0.0, type=float, help='')
    ap.add_argument('--num_beam_groups', required=False, default=1, type=int, help='')

    return ap.parse_args()

class Model:

    def __init__(self, args):

        self.tokenizer = T5Tokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            use_fast=True        
        )

        self.model = MT5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
        )

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        self.prefix = args.prefix
        self.args = args

    def prepare_texts(self, texts):
        return [self.prefix + text for text in texts]

    def generate(self, texts):
    
        texts = self.prepare_texts(texts) # add prefix
        inputs = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        generated_ids = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            use_cache=True,
            max_length=self.args.max_output_length if self.args.max_output_length is not None else 10,
            num_beams=self.args.beam_size, 
            # early_stopping=True,
            do_sample=self.args.do_sample,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            num_return_sequences=self.args.num_return_sequences,
            num_beam_groups=self.args.num_beam_groups,
            diversity_penalty=self.args.diversity_penalty,

            )

        return self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)



if __name__ == "__main__":

    args = set_args()
    
    model = Model(args)

    input_texts = [
        """Das Bundesgericht hat entschieden, dass ausserhalb der Bauzonen erstellte Bauten, wie Tessiner Rustici oder Maiensässe illegal sind. Das heisst: Auch die vor über 30 Jahren illegal erstellten Bauten müssten eigentlich abgerissen werden. Wie viele es sind, kann man nur schätzen. Tausende Häuser, vor allem Rustici und Maiensässe in den Gebirgskantonen Tessin, Graubünden, Bern und Wallis, sind laut «Tages-Anzeiger» potenziell von diesem Urteilsspruch betroffen. Doch nun könnte es ein Happy End für die Besitzer von Häusern geben, die ausserhalb der Bauzone erstellt wurden. Am Donnerstag wurde das Thema im Nationalrat behandelt. Eine Motion sieht vor, dass eine Verjährungsfrist 30 Jahre nach dem Bau eintritt. Kehrtwende war ein Bundesgerichtsentscheid vom April 2021. Es beurteilte Gebäude einer Luzerner Baufirma als rechtswidrig. Die Gebäude wurden über Jahre hinweg illegal erstellt. Bisher ging man von einer Verjährungsfrist von 30 Jahren aus, das Bundesgericht entschied aber anders. Dies geht dem St. Galler SVP-Nationalrat Mike Egger gegen den Strich. «Wenn eine Gemeinde 30 Jahre nicht gemerkt habe, dass ein Bau illegal ist, dann solle die Verjährung greifen. Diese Frist ist mehr als genug Zeit, um solche Bauten zu entdecken und abreissen zu lassen», sagt Egger, der das Vorhaben in die Raumplanungskommission gebracht hat. Er befürchtet viele Rechtsstreitereien, wenn das Bundesgesetz durchgesetzt werden muss. Andere, wenige Kommissionsmitglieder finden, dass die Regel nicht geändert werden darf, das würde Hausbesitzer belohnen, die illegal handelten, sagen sie. Im Tessin gibt es laut SRF an die 2000 Rustici, die juristisch anfechtbar ausserhalb der Bauzonen errichtet wurden, schätzt Baudirektor Claudio Zali. Nun ist das Parlament gefragt. Entscheidet es, dass es zu kompliziert ist, den Bundesgerichtsentscheid umzusetzen und die illegal gebauten Rustici und Chalets abzureissen, dürften sie bestehen bleiben.""",
        """News-Scout B.* erhielt einen Brief der Regionalwerke Baden. Sie war schockiert, als sie las, dass eine massive Anpassung des Energiepreises ab April in der Grössenordnung einer Verdopplung eintreten könnte (siehe Bildstrecke oben). Sie fragte sich, ob sie dann die Heizung abstellen soll. Solche Briefe dürften einige Schweizer Haushalte bekommen. Die Hälfte der Liegenschaften in der Schweiz heizt noch mit Öl und Gas, sagt Comparis-Finanzexperte Michael Kuhn zu 20 Minuten. Wie teuer es wird, hängt davon ab, ob das Öl und Gas mit langfristigen Verträgen zu tiefen Preisen oder teuer an der Rohstoffbörse bezogen wird. Grundsätzlich könnten die Energiewerke die Preise eins zu eins weitergeben. «Wenn der Ölpreis um 100 Prozent steigt, könnten sich auch die Heizkosten verdoppeln», so Kuhn. Entweder kommt der Nebenkosten-Schock sofort bei einer Konto-Abrechnung oder mit einer Pauschale etwas später. Der Comparis-Experte empfiehlt angesichts der hohen Energiepreise, etwas Geld für die Nebenkosten zurückzulegen. Damit soll man bei der nächsten Abrechnung gewappnet sein vor bösen Überraschungen. Im Sommer brauchts nur noch Energie fürs Warmwasser und nicht mehr für die Heizung, sagt Robert Weinert, Leiter Immo-Monitoring bei Wüest Partner, zu 20 Minuten. In den kälteren Bergregionen und in schlecht isolierten Häusern brauchts bei tiefen Temperaturen aber auch im Sommer eine Heizung. Mieter und Mieterinnen in älteren Liegenschaften könnten deshalb stark davon profitieren, wenn es einen Umstieg von der Öl- oder Gasheizung zu einer Wärmepumpe gebe. Viele Vermietern Vermieterinnen würden jetzt umsteigen. Durch die Wertsteigerung könne die Miete zwar kurzfristig steigen, am Ende liessen sich aber mehr Nebenkosten sparen, so Weinert. Allerdings werden Vermietende, die in den vergangenen Jahren eine neue Ölheizung installieren liessen, wohl kaum aufgrund der derzeitigen Situation schon wieder umbauen. «Das wäre nicht nachhaltig und wird vom Gesetzgeber auch nicht vorgeschrieben», sagt Thomas Oberle vom Hauseigentümerverband Schweiz. Bei dem Absender der Briefe, den Regionalwerken Baden, heisst es auf Anfrage, dass die definitive Erhöhung erst per 1. April bekannt gegeben werde. Man habe mit dem Brief an die Kundinnen und Kunden über die extreme Preissituation am Gasmarkt informieren wollen. Falls es zur Verdopplung der Energiepreise käme, müsste ein Einfamilienhaus laut Marketingleiter Gilles Tornare bei einem üblichen Jahresverbrauch von 20'000 kWh rund 150 Franken pro Monat mehr zahlen.""",
    ]

    output_texts = model.generate(input_texts)

    output_texts = np.array_split(np.array(output_texts), len(input_texts))

    for input_text, output_text in zip(input_texts, output_texts):
        print('****')
        print('Input:', input_text) 
        for i, o in enumerate(output_text):         
            print(f'Ouput {i}:', output_text)
        print('****')


    
