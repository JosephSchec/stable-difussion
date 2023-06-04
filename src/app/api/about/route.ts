import { HfInference } from '@huggingface/inference'
import axios from 'axios';
import { readFileSync } from 'fs'
import { NextResponse } from 'next/server';



const TOKEN=process.env.HUGFACE_TOKEN 
const hf = new HfInference( TOKEN,{wait_for_model:true,use_cache:false,retry_on_error:false,use_gpu:true})
// Natural Language
export async function GET() {
console.log(process.env.HUGFACE_TOKEN, TOKEN)


// await hf.summarization({
//   model: 'facebook/bart-large-cnn',
//   inputs:
//     'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930.',
//   parameters: {
//     max_length: 100
//   }
// })

// await hf.questionAnswering({
//   model: 'deepset/roberta-base-squad2',
//   inputs: {
//     question: 'What is the capital of France?',
//     context: 'The capital of France is Paris.'
//   }
// })

// await hf.tableQuestionAnswering({
//   model: 'google/tapas-base-finetuned-wtq',
//   inputs: {
//     query: 'How many stars does the transformers repository have?',
//     table: {
//       Repository: ['Transformers', 'Datasets', 'Tokenizers'],
//       Stars: ['36542', '4512', '3934'],
//       Contributors: ['651', '77', '34'],
//       'Programming language': ['Python', 'Python', 'Rust, Python and NodeJS']
//     }
//   }
// })

// await hf.textClassification({
//   model: 'distilbert-base-uncased-finetuned-sst-2-english',
//   inputs: 'I like you. I love you.'
// })

// const a=await hf.textGeneration({
//   model: 'gpt2',
//   inputs: 'The answer to the universe is'
// })
// console.log(a)


// await hf.tokenClassification({
//   model: 'dbmdz/bert-large-cased-finetuned-conll03-english',
//   inputs: 'My name is Sarah Jessica Parker but you can call me Jessica'
// })

// await hf.translation({
//   model: 't5-base',
//   inputs: 'My name is Wolfgang and I live in Berlin'
// })

// await hf.zeroShotClassification({
//   model: 'facebook/bart-large-mnli',
//   inputs: [
//     'Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!'
//   ],
//   parameters: { candidate_labels: ['refund', 'legal', 'faq'] }
// })

// await hf.conversational({
//   model: 'microsoft/DialoGPT-large',
//   inputs: {
//     past_user_inputs: ['Which movie is the best ?'],
//     generated_responses: ['It is Die Hard for sure.'],
//     text: 'Can you explain why ?'
//   }
// })

// await hf.sentenceSimilarity({
//   model: 'sentence-transformers/paraphrase-xlm-r-multilingual-v1',
//   inputs: {
//     source_sentence: 'That is a happy person',
//     sentences: [
//       'That is a happy dog',
//       'That is a very happy person',
//       'Today is a sunny day'
//     ]
//   }
// })

// await hf.featureExtraction({
//   model: "sentence-transformers/distilbert-base-nli-mean-tokens",
//   inputs: "That is a happy person",
// });


// // Computer Vision

const prompt=`Generate an entertaining image depicting Cookie Monster in a gym setting, showcasing his enthusiasm for fitness and indulgence in cookies. Picture Cookie Monster at a vibrant and energetic gym, surrounded by exercise equipment. He stands in the center of the frame, displaying his characteristic blue fur and wide grin. Cookie Monster wears a sweatband around his forehead, signaling his commitment to the workout. His workout attire consists of a comfortable and stylish gym outfit, complete with shorts and a t-shirt with a cookie-themed design. In one hand, he holds a dumbbell, showcasing his dedication to building strength, while the other hand holds a tray filled with his favorite cookies. The gym environment should feature bright colors, motivational posters, and other gym enthusiasts engaged in various exercises, capturing the energetic atmosphere. Cookie Monster's expression should convey a mix of determination and excitement as he balances his love for cookies with his fitness goals. The image should capture the fun-loving spirit
 of Cookie Monster while highlighting his passion for staying active and enjoying his favorite treat in moderation.`

const img=await hf.textToImage({
  inputs:prompt,
  model: 'runwayml/stable-diffusion-v1-5',
  parameters:{height:512,width:512,
   negative_prompt:"fuzzy"
}
});
 


const {type } = img

const result = await img.arrayBuffer() 

const base64data = Buffer.from(result).toString('base64')

const base64 = `data:${type};base64,` + base64data

  return new Response(base64) ; 

// await hf.zeroShotImageClassification({
//   model: 'openai/clip-vit-large-patch14-336',
//   inputs: {
//     image: await (await fetch('https://placekitten.com/300/300')).blob()
//   },  
//   parameters: {
//     candidate_labels: ['cat', 'dog']
//   }
// })

// // Multimodal

// await hf.visualQuestionAnswering({
//   model: 'dandelin/vilt-b32-finetuned-vqa',
//   inputs: {
//     question: 'How many cats are lying down?',
//     image: await (await fetch('https://placekitten.com/300/300')).blob()
//   }
// })

// await hf.documentQuestionAnswering({
//   model: 'impira/layoutlm-document-qa',
//   inputs: {
//     question: 'Invoice number?',
//     image: await (await fetch('https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png')).blob(),
//   }
// })

// // Tabular

// await hf.tabularRegression({
//   model: "scikit-learn/Fish-Weight",
//   inputs: {
//     data: {
//       "Height": ["11.52", "12.48", "12.3778"],
//       "Length1": ["23.2", "24", "23.9"],
//       "Length2": ["25.4", "26.3", "26.5"],
//       "Length3": ["30", "31.2", "31.1"],
//       "Species": ["Bream", "Bream", "Bream"],
//       "Width": ["4.02", "4.3056", "4.6961"]
//     },
//   },
// })

// await hf.tabularClassification({
//   model: "vvmnnnkv/wine-quality",
//   inputs: {
//     data: {
//       "fixed_acidity": ["7.4", "7.8", "10.3"],
//       "volatile_acidity": ["0.7", "0.88", "0.32"],
//       "citric_acid": ["0", "0", "0.45"],
//       "residual_sugar": ["1.9", "2.6", "6.4"],
//       "chlorides": ["0.076", "0.098", "0.073"],
//       "free_sulfur_dioxide": ["11", "25", "5"],
//       "total_sulfur_dioxide": ["34", "67", "13"],
//       "density": ["0.9978", "0.9968", "0.9976"],
//       "pH": ["3.51", "3.2", "3.23"],
//       "sulphates": ["0.56", "0.68", "0.82"],
//       "alcohol": ["9.4", "9.8", "12.6"]
//     },
//   },
// })


// // Using your own inference endpoint: https://hf.co/docs/inference-endpoints/
// const gpt2 = hf.endpoint('https://xyz.eu-west-1.aws.endpoints.huggingface.cloud/gpt2');
// const { generated_text } = await gpt2.textGeneration({inputs: 'The answer to the universe is'});
}
