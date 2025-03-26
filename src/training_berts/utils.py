from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from torch import where, cat, ones, zeros, sigmoid
import torchmetrics

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def make_predictions(model, data_iter, device, problem: str = "single", threshold: int = 0.5):
        true, preds = [], []
        #for each batch add predictions and correct labels
        for batch in data_iter:
            #send to gpu if available
            batch = batch.to(device)
            #predict labels with the model
            outputs = model(**batch)

            #get the most probable label
            if problem == "single":
                predictions = outputs["logits"].argmax(dim = 1)
            else:
                predictions = where(sigmoid(outputs["logits"]) < threshold, 0, 1)
            #get the gold labels
            labels = batch["labels"]

            #add each element of the filtered labels and the filtered predictions to their respective lists
            #true.extend(labels)
            #preds.extend(predictions)
            true.append(labels.cpu())
            preds.append(predictions.cpu())

        return cat(true),cat(preds)
        #return true, preds

#report scores
def report(model, data_iter, int_to_label, label_vocab, device):
    #get predictions and gold labels using the above function
    y_true, y_preds = make_predictions(model, data_iter, device)
    preds = [int_to_label[int.item()] for int in y_preds]
    gold = [int_to_label[int.item()] for int in y_true]
    #report score metrics
    print("--------------------------------------")
    print(f"Evaluation")
    #f1 score using strict evaluation
    #print(f"F1: {f1_score(y_true, y_preds, zero_division = 0, average = 'micro'):.3f}")
    #classification report with precision, recall, f1 for each tag using strict evaluation
    print(f"\nClassification Report:")
    print(classification_report(preds, gold, zero_division = 0, labels = label_vocab))
    #confusion matrix
    print(f"\nConfusion Matrix:")
    print_cm(confusion_matrix(preds, gold, labels = label_vocab), label_vocab)


def multi_report(model, data_iter, int_to_label, label_vocab, device):
    y_true, y_preds = make_predictions(model, data_iter, device, problem = "multi")
    #report score metrics
    print("--------------------------------------")
    print(f"Evaluation")
    #f1 score using strict evaluation
    #print(f"F1: {f1_score(y_true, y_preds, zero_division = 0, average = 'micro'):.3f}")
    #classification report with precision, recall, f1 for each tag using strict evaluation
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_preds, zero_division = 0, target_names = label_vocab))


def test(model, tokenizer, data_set, int_to_label, threshold: int = 0.5):
    #y_true, y_preds = make_predictions(model, data_iter)

    supported_languages = list(int_to_label.values())
    #language_to_index = {language: i for i, language in enumerate(supported_languages)}

    # loose metrics
    loose_accuracy = torchmetrics.Accuracy("binary")
    loose_per_language_f1 = {language: torchmetrics.F1Score("binary") for language in supported_languages}
    loose_per_language_mcc = {language: torchmetrics.MatthewsCorrCoef("binary") for language in supported_languages}

    # strict metrics
    strict_accuracy = torchmetrics.Accuracy("binary")
    overlap_f1 = 0.0
    strict_per_language_f1 = {language: torchmetrics.F1Score("binary", ) for language in supported_languages}
    strict_per_language_mcc = {language: torchmetrics.MatthewsCorrCoef("binary") for language in supported_languages}


    for sample in samples:
        text = sample["text"]
        sample["gold_languages"] = set(sample["languages"])
        
        input_ids = tokenizer(text, return_tensors = "pt", padding = True)
        pred = model(**input_ids)["logits"]
        sample["predicted_languages"] = set([int_to_label[i.item()] for i in where(pred[0] > threshold)[0]])


    print("Calculating metrics...")

    for sample in samples:
        gold_languages = sample["gold_languages"]
        predicted_languages = sample["predicted_languages"]

        # loose metrics
        if predicted_languages.issubset(gold_languages):
            loose_accuracy.update(ones(1), ones(1))

            for language in supported_languages:
                if language in predicted_languages and language in gold_languages:
                    loose_per_language_f1[language].update(ones(1), ones(1))
                    loose_per_language_mcc[language].update(ones(1), ones(1))
                elif language not in predicted_languages and language not in gold_languages:
                    loose_per_language_f1[language].update(zeros(1), zeros(1))
                    loose_per_language_mcc[language].update(zeros(1), zeros(1))
        else:
            loose_accuracy.update(zeros(1), ones(1))

            for language in supported_languages:
                if language in predicted_languages and language in gold_languages:
                    loose_per_language_f1[language].update(ones(1), zeros(1))
                    loose_per_language_mcc[language].update(ones(1), zeros(1))
                elif language in predicted_languages:
                    loose_per_language_f1[language].update(zeros(1), ones(1))
                    loose_per_language_mcc[language].update(zeros(1), ones(1))
                elif language in gold_languages:
                    loose_per_language_f1[language].update(zeros(1), ones(1))
                    loose_per_language_mcc[language].update(zeros(1), ones(1))
                else:
                    loose_per_language_f1[language].update(zeros(1), zeros(1))
                    loose_per_language_mcc[language].update(zeros(1), zeros(1))

        # strict metrics
        if predicted_languages == gold_languages:
            strict_accuracy.update(ones(1), ones(1))

        else:
            strict_accuracy.update(zeros(1), ones(1))
        
        common_languages = len(predicted_languages.intersection(gold_languages))
        overlap_precision = common_languages / len(predicted_languages)
        overlap_recall = common_languages / len(gold_languages)
        if overlap_precision + overlap_recall > 0:
            overlap_f1 += 2 * overlap_precision * overlap_recall / (overlap_precision + overlap_recall)

        for language in supported_languages:
            if language in predicted_languages and language in gold_languages:
                strict_per_language_f1[language].update(ones(1), ones(1))
                strict_per_language_mcc[language].update(ones(1), ones(1))
            elif language in predicted_languages:
                strict_per_language_f1[language].update(ones(1), zeros(1))
                strict_per_language_mcc[language].update(ones(1), zeros(1))
            elif language in gold_languages:
                strict_per_language_f1[language].update(zeros(1), ones(1))
                strict_per_language_mcc[language].update(zeros(1), ones(1))
            else:
                strict_per_language_f1[language].update(zeros(1), zeros(1))
                strict_per_language_mcc[language].update(zeros(1), zeros(1))


    # pretty print the confusion matrix
    print(f"\n# Results:\n")

    print("## Loose metrics")
    
    print(f"\tLoose accuracy: {loose_accuracy.compute().item():.2%}")
    print(f"\tLoose macro F1: {sum([loose_per_language_f1[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print(f"\tLoose macro MCC: {sum([loose_per_language_mcc[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print()
    print("### Per-language metrics")
    for language in supported_languages:
        print(f"\t{language}:")
        print(f"\t\tF1: {loose_per_language_f1[language].compute().item():.2%}")
        print(f"\t\tMCC: {loose_per_language_mcc[language].compute().item():.2%}")


    print("\n\n## Strict metrics")
    print(f"\tStrict accuracy: {strict_accuracy.compute().item():.2%}")
    print(f"\tOverlap F1: {overlap_f1 / len(samples):.2%}")
    print(f"\tStrict macro F1: {sum([strict_per_language_f1[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print(f"\tStrict macro MCC: {sum([strict_per_language_mcc[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print()
    print("### Per-language metrics")
    for language in supported_languages:
        print(f"\t{language}:")
        print(f"\t\tF1: {strict_per_language_f1[language].compute().item():.2%}")
        print(f"\t\tMCC: {strict_per_language_mcc[language].compute().item():.2%}")