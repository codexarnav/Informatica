from transformers import pipeline

pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
def sentiment_analysis(contents):
  results =[]
  for cont in contents:
    content = cont[:512]
    result= pipe(content)[0]
    results.append({

          'text': cont,
          'label': result['label'],
          'score': result['score']
      })

  sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
  return sorted_results
