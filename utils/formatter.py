def get_unique_targets(targets):
	return list(set(map(lambda label: label[0].split('-')[-1], list(targets))))

def flatten_prediction_results(data_bundle, databundle_for_test, subset, predicted_entities):
	#print(list(databundle_for_test.get_dataset(subset).get_field('chars')))
    targets = data_bundle.vocabs["target"]
    #
    chars = data_bundle.vocabs["chars"]
    #predicted_entities = self._predict(subset_for_prediction = databundle_for_test.get_dataset(subset), targets = targets, filename = None)
    #print(predicted_entities)
    #print()
    true_entities = list(map(lambda sentence: [targets.to_word(label).split("-")[-1] for label in sentence], list(databundle_for_test.get_dataset(subset).get_field('target'))))
    true_entity_words = list(map(lambda sentence: [chars.to_word(label) for label in sentence], list(databundle_for_test.get_dataset(subset).get_field('chars'))))
    #print(dir(databundle_for_test.get_dataset(subset)))
    #print(true_entity_words)
    predicted_entity_index = 0
    true_entity_index = 0
    
    flattened_predicted_entities = []
    flattened_true_entities = []

    for sentence, sentence_words in zip(true_entities, true_entity_words):
        true_entity_index = 0
        previous_sentence_word = None
        for label in sentence:
            flattened_predicted_entities.append(predicted_entities[predicted_entity_index].split('\t')[1])
            #print(label)
            flattened_true_entities.append(label)
            # if len(predicted_entities[predicted_entity_index].split('\t')) < 2:
            # 	print("*"*100)
            # print(f'{label} {sentence_words[true_entity_index]} | {predicted_entities[predicted_entity_index]}')
            # if sentence_words[true_entity_index] != previous_sentence_word:
            predicted_entity_index += 1
            previous_sentence_word = sentence_words[true_entity_index]
            true_entity_index += 1
        #print(sentence_words[true_entity_index-1], sentence_words[true_entity_index-2])
        if sentence_words[true_entity_index-1] != sentence_words[true_entity_index-2]:
            predicted_entity_index += 1
        else:
            predicted_entity_index -= 1
            #print('*'*100)
    return flattened_true_entities, flattened_predicted_entities