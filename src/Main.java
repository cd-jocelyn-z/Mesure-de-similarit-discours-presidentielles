import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.Map.Entry;
import java.text.DecimalFormat;

public class Main {
    public static void main(String[] args) throws IOException {
        HashMap<String, String> speechDictionary = getSpeechDictionary("US_Inaugural_Addresses");
        HashMap<String, ArrayList<String>> segmentedSpeechDictionary = getSegmentSpeechDictionaryValue(speechDictionary);
        HashMap<String,ArrayList<String>> nMostSimilarDocs = getNMostSimilarDocs(segmentedSpeechDictionary, 3);
        ArrayList<String> termsInCommonFromOldestRecent = getTermsInCommonFromOldestRecent(segmentedSpeechDictionary);
        ArrayList <String> docTerms = getDocTerms(segmentedSpeechDictionary);
        HashMap<String, Integer> termFrequency = getTermFrequency(docTerms);
        HashMap<String, Double> documentFrequency = getDocumentFrequency(segmentedSpeechDictionary);
        String[] targetTerms = new String[]{"government", "borders", "people", "obama", "war", "honor", "foreign", "men", "women", "children"};
        HashMap<String, HashMap<String, Double>> targetTermsImportance = getTfIdfOfTargetTerms(targetTerms, segmentedSpeechDictionary);
        getPrintTfIdfOfTargetTerms(targetTerms, targetTermsImportance, segmentedSpeechDictionary);
        HashMap <String, ArrayList<String>> nMostImportantWords = getNMostImportantTerms(segmentedSpeechDictionary, 5);
        HashMap <String, HashMap<String, Double>> tfIdfForAllDocsMap = getTfIdf(segmentedSpeechDictionary);
        HashMap<String, HashMap<String, Double>> cosineSimilarity = getCosineSimilarity(segmentedSpeechDictionary);
    }

    public static HashMap<String, String> getSpeechDictionary(String speechFolderName) throws IOException {
        HashMap<String, String> speechDictionary = new HashMap<>();

        Path speechFolderPath = Paths.get(speechFolderName);
        if (Files.exists(speechFolderPath)) {
            File[] speechFilesList = speechFolderPath.toFile().listFiles();

            for (File speechFileName : speechFilesList) {
                String fileName = speechFileName.getName();
                String speechText = new String(Files.readAllBytes(Paths.get(speechFileName.getAbsolutePath())));

                if (!speechDictionary.containsKey(fileName)) {
                    speechDictionary.put(fileName, speechText);
                }
            }
        }
        return speechDictionary;
    }

    public static HashMap<String, ArrayList<String>> getSegmentSpeechDictionaryValue(HashMap<String, String> speechDictionary) {
        HashMap<String, ArrayList<String>> speechDictionaryWithSegmentedValue = new HashMap<>();

        for(Entry<String, String> entry : speechDictionary.entrySet()) {
            String fileName = entry.getKey();
            String rawText = entry.getValue();
            String reworkedText = "";

            ArrayList<String> segmentedText = new ArrayList<>(Arrays.asList(rawText.split(" ")));
            for (String element : segmentedText) {
                int[] elementCodePoints = element.codePoints().toArray();

                ArrayList<Integer> codePointsList = new ArrayList<>();
                for (int codePoint : elementCodePoints) {
                    codePointsList.add(codePoint);
                }

                int codePointIndex = 0;
                while( codePointIndex<= codePointsList.size() - 1) {
                    int codePoint =  codePointsList.get(codePointIndex);

                    if (!(Character.isDigit(codePoint) || Character.isAlphabetic(codePoint))) {
                        codePointsList.add(codePointIndex + 1,32);
                        codePointsList.add(codePointIndex,32);
                        codePointIndex += 2;
                    }
                    codePointIndex += 1;
                }

                for (int codePoint : codePointsList) {
                    reworkedText += Character.toString(codePoint);
                }

                reworkedText += " ";
            }

            ArrayList<String> tokenizedTextList = new ArrayList<>();
            for(String token : reworkedText.toLowerCase().split(" ")) {
                tokenizedTextList.add(token);
            }

            speechDictionaryWithSegmentedValue.put(fileName, tokenizedTextList);
        }

        return speechDictionaryWithSegmentedValue;
    }

    public static HashMap<String,ArrayList<String>> getNMostSimilarDocs (HashMap<String, ArrayList<String>> segmentedSpeechDictionary, Integer nMostSimilar) {
        HashMap<String,ArrayList<String>> nMostSimilarDocs = new HashMap<>();
        HashMap <String,HashMap<String, Integer>> docSimilarityMap = new HashMap<>();

        for(Entry <String, ArrayList<String>> entryA : segmentedSpeechDictionary.entrySet()) {
            String referenceDoc = entryA.getKey();
            ArrayList <String> refTokenizedText = entryA.getValue();

            for(Entry <String, ArrayList<String>> entryB : segmentedSpeechDictionary.entrySet()) {
                String comparedDoc = entryB.getKey();
                ArrayList<String> compTokenizedText = entryB.getValue();

                if (!referenceDoc.equals(comparedDoc)) {
                    HashSet<String> uniqueRefTokens = new HashSet<>(refTokenizedText);
                    HashSet<String> uniqueCompTokens = new HashSet<>(compTokenizedText);

                    int n = uniqueRefTokens.size();
                    uniqueRefTokens.removeAll(uniqueCompTokens);
                    int similarityScore = n - uniqueRefTokens.size();

                    if (docSimilarityMap.containsKey(referenceDoc)) {
                        docSimilarityMap.get(referenceDoc).put(comparedDoc, similarityScore);
                    } else {
                        HashMap<String, Integer> docSimilarityScoreMap = new HashMap<>();
                        docSimilarityScoreMap.put(comparedDoc, similarityScore);
                        docSimilarityMap.put(referenceDoc, docSimilarityScoreMap);
                    }
                }
            }
        }

        for (Entry <String,HashMap<String, Integer>> outerMapEntry : docSimilarityMap.entrySet()) {
            String referenceDoc = outerMapEntry.getKey();
            HashMap <String, Integer> docSimilarityScoresMap = outerMapEntry.getValue();

            ArrayList <Integer> scoresList = new ArrayList<>();
            for (Integer similarityScore : docSimilarityScoresMap.values()) {
                scoresList.add(similarityScore);
            }
            Collections.sort(scoresList, Comparator.reverseOrder());

            ArrayList <String> sortedSimilarDocList = new ArrayList<>();
            for(Integer similarityScore : scoresList) {
                for (Entry <String, Integer> innerMapEntry : docSimilarityScoresMap.entrySet()) {
                    String comparedDoc = innerMapEntry.getKey();
                    Integer score = innerMapEntry.getValue();

                    if(similarityScore.equals(score) && sortedSimilarDocList.size() < nMostSimilar) {
                        sortedSimilarDocList.add(comparedDoc);
                    }
                }
            }
            nMostSimilarDocs.put(referenceDoc, sortedSimilarDocList);
        }
        return nMostSimilarDocs;
    }

    public static ArrayList<String> getTermsInCommonFromOldestRecent(HashMap<String, ArrayList<String>>segmentedSpeechDict ) {
        ArrayList<String> commonTermsBetweenOldestAndRecent = new ArrayList<>();
        ArrayList<Integer> listOfYears = new ArrayList<>();

        for(Entry<String, ArrayList<String>> entry : segmentedSpeechDict.entrySet()) {
            String fileName = entry.getKey().replace(".txt","");
            String[] fileNameSegments = fileName.split("_");
            String yearPart = fileNameSegments[fileNameSegments.length -1];
            int year = Integer.parseInt(yearPart);
            listOfYears.add(year);
        }
        Collections.sort(listOfYears);

        int oldestYear = listOfYears.get(0);
        int recentYear = listOfYears.get(listOfYears.size() - 1);
        String oldestYearFound = Integer.toString(oldestYear);
        String recentYearFound = Integer.toString(recentYear);
        HashSet <String> uniqueTermsFromOldest = new HashSet<>();
        HashSet <String> uniqueTermsFromRecent = new HashSet<>();
        for (Entry<String, ArrayList<String>> entry : segmentedSpeechDict.entrySet()) {
            String fileName = entry.getKey().replace(".txt", "");
            boolean isOldestFile = fileName.contains(oldestYearFound);
            boolean isRecentFile = fileName.contains(recentYearFound);

            if(isOldestFile) {
                ArrayList<String> contentFromOldest = entry.getValue();
                uniqueTermsFromOldest.addAll(contentFromOldest);
            }
            if(isRecentFile) {
                ArrayList<String> contentFromRecent = entry.getValue();
                uniqueTermsFromRecent.addAll(contentFromRecent);
            }
        }

        for(String uniqueTermFromOldest : uniqueTermsFromOldest) {
            for(String uniqueTermFromRecent : uniqueTermsFromRecent) {
                if(uniqueTermFromOldest.equals(uniqueTermFromRecent)) {

                    if(!commonTermsBetweenOldestAndRecent.contains(uniqueTermFromOldest)) {
                        commonTermsBetweenOldestAndRecent.add(uniqueTermFromOldest);
                    }
                }
            }
        }
        return commonTermsBetweenOldestAndRecent;
    }

    public static ArrayList<String> getDocTerms(HashMap<String, ArrayList<String>> segmentedSpeechDictionary) {
        ArrayList<String> docTerms = new ArrayList<>();

        for(ArrayList<String> tokenizedTextList : segmentedSpeechDictionary.values()){
            for(String term : tokenizedTextList) {
                docTerms.add(term);
            }
        }
        return docTerms;
    }

    public static HashMap<String, Integer> getTermFrequency(ArrayList<String> docTerms) {
        HashMap<String, Integer> termFreq = new HashMap<>();

        HashSet <String> uniqueTerms = new HashSet<>();
        uniqueTerms.addAll(docTerms);
        for(String uniqueTerm : uniqueTerms) {
            int termOcc = 0;
            for (String term : docTerms) {
                if (uniqueTerm.equals(term)){
                    termOcc ++;
                }
            }
            termFreq.put(uniqueTerm, termOcc);
        }
        return termFreq;
    }

    public static HashMap<String, Double> getDocumentFrequency(HashMap <String,ArrayList<String>> segmentedSpeechDictionary) {
        HashMap<String, Double> docFreqMap = new HashMap<>();

        HashSet<String> uniqueWordsGrandCollection = new HashSet<>();
        for (ArrayList<String> tokenizedTextList : segmentedSpeechDictionary.values()) {
            uniqueWordsGrandCollection.addAll(tokenizedTextList);
        }

        for (String term : uniqueWordsGrandCollection) {
            double numDocsContainingTerm = 0.0;

            for(ArrayList<String> tokenizedTextList : segmentedSpeechDictionary.values()) {
                if (tokenizedTextList.contains(term)) {
                    numDocsContainingTerm++;
                }
            }
            double docFreq = numDocsContainingTerm / segmentedSpeechDictionary.size();
            docFreqMap.put(term,docFreq);
        }
        return docFreqMap;
    }

    public static HashMap<String, HashMap<String, Double>> getTfIdfOfTargetTerms(String [] targetTerms, HashMap <String,ArrayList<String>> segmentedSpeechDictionary) {
        HashMap<String, HashMap<String, Double>> tfIdfOfTargetTerms = new HashMap<>();
        HashMap<String, Double> documentFrequencies = getDocumentFrequency(segmentedSpeechDictionary);

        for(Entry<String, ArrayList<String>> entry : segmentedSpeechDictionary.entrySet()) {
            String docName = entry.getKey();
            ArrayList<String> tokenizedTextList = entry.getValue();

            for(String targetTerm : targetTerms) {
                if(tokenizedTextList.contains(targetTerm)) {
                    HashMap<String, Integer> tf = getTermFrequency(tokenizedTextList);
                    double termFreq = tf.get(targetTerm);
                    Double docFreq = documentFrequencies.get(targetTerm);

                    double N = segmentedSpeechDictionary.size();
                    double IDF = Math.log(N  / docFreq) + 1;

                    double tfIdf = termFreq * IDF;

                    if(tfIdfOfTargetTerms.containsKey(docName)) {
                        tfIdfOfTargetTerms.get(docName).put(targetTerm, tfIdf);
                    }else{
                        HashMap<String,Double> newInnerMap = new HashMap<>();
                        newInnerMap.put(targetTerm,tfIdf);
                        tfIdfOfTargetTerms.put(docName, newInnerMap);
                    }
                }
            }
        }
        return tfIdfOfTargetTerms;
    }

    public static LinkedHashMap<String, LinkedHashMap<String, Double>> getPrintTfIdfOfTargetTerms(String [] targetTerms, HashMap<String, HashMap<String, Double>> targetTermsImportance, HashMap<String, ArrayList<String>> segmentedSpeechDictionary) {
        LinkedHashMap<String, LinkedHashMap<String, Double>> printTfIdfOfTargetTerms = new LinkedHashMap<>();
        DecimalFormat decimalLimit = new DecimalFormat("#.##");

        ArrayList <Integer> listOfYears = new ArrayList<>();
        for(Entry<String, ArrayList<String>> entry : segmentedSpeechDictionary.entrySet()) {
            String fileName = entry.getKey().replace(".txt","");
            String[] fileNameSegments = fileName.split("_");
            String yearPart = fileNameSegments[fileNameSegments.length -1];
            int year = Integer.parseInt(yearPart);
            listOfYears.add(year);
        }
        Collections.sort(listOfYears);

        for (Integer year : listOfYears) {
              for(String entry : targetTermsImportance.keySet()) {
                  String fileName = entry;

                  if(fileName.contains(year.toString())) {
                    printTfIdfOfTargetTerms.put(fileName, new LinkedHashMap<>());
                  }
            }
        }

        ArrayList<Double> sortedTermsScore = new ArrayList<>();
        for(String term : targetTerms) {
            for (Entry<String, HashMap<String, Double>> outerMap : targetTermsImportance.entrySet()) {
                for (Entry<String, Double> innerMapEntry : outerMap.getValue().entrySet()) {
                    String targetTerm = innerMapEntry.getKey();
                    Double similarityScore = innerMapEntry.getValue();

                    if (term.equals(targetTerm)) {
                        sortedTermsScore.add(similarityScore);
                    }
                }
            }
        }

        sortedTermsScore.sort(Comparator.reverseOrder());
        for(Double score : sortedTermsScore) {
            for (Entry<String, HashMap<String, Double>> outerMapEntry : targetTermsImportance.entrySet()) {
                String docName = outerMapEntry.getKey();
                HashMap<String, Double> tfIdfMap = outerMapEntry.getValue();
                for(Entry <String, Double> entry : tfIdfMap.entrySet()) {
                    String targetTerm = entry.getKey();
                    Double similarityScore = entry.getValue();

                    if(similarityScore.equals(score)) {
                        printTfIdfOfTargetTerms.get(docName).put(targetTerm, Double.parseDouble(decimalLimit.format(similarityScore)));
                    }
                }
            }
        }

        return printTfIdfOfTargetTerms;
    }

    public static HashMap <String, ArrayList<String>> getNMostImportantTerms(HashMap <String,ArrayList<String>> segmentedSpeechDictionary, Integer nMostImportant){
        HashMap <String, ArrayList<String>> nMostImportantTerms = new HashMap<>();
        HashMap <String, HashMap<String, Double>> nMostImportantScores = new HashMap<>();

        HashMap<String, Double> documentFrequencies = getDocumentFrequency(segmentedSpeechDictionary);

        for(Entry<String, ArrayList<String>> entry : segmentedSpeechDictionary.entrySet()) {
            String docName = entry.getKey();
            ArrayList<String> tokenizedTextList = entry.getValue();

            HashMap<String, Integer> tf = getTermFrequency(tokenizedTextList);
            for(Entry <String, Integer> tfEntry : tf.entrySet()) {
                String term = tfEntry.getKey();
                Integer termFreq = tfEntry.getValue();

                Double docFreq = documentFrequencies.get(term);

                double N = segmentedSpeechDictionary.size();
                double IDF = Math.log(N  / docFreq) + 1;

                double tfIdf = termFreq * IDF;

                if(nMostImportantScores.containsKey(docName)) {
                    nMostImportantScores.get(docName).put(term, tfIdf);
                }else{
                    HashMap<String,Double> newInnerMap = new HashMap<>();
                    newInnerMap.put(term,tfIdf);
                    nMostImportantScores.put(docName, newInnerMap);
                }
            }
        }

        for(Entry <String, HashMap<String, Double>> outerMapEntry : nMostImportantScores.entrySet()) {
            String docName = outerMapEntry.getKey();
            HashMap<String, Double> termScoreMap = outerMapEntry.getValue();
            ArrayList <Double> scoreList = new ArrayList<>();
            for(Entry <String, Double> entry : termScoreMap.entrySet()) {
                Double innerMapValue = entry.getValue();
                scoreList.add(innerMapValue);
            }
            Collections.sort(scoreList, Comparator.reverseOrder());

            ArrayList <String> sortedImportantTermsList = new ArrayList<>();
            HashSet<Double> scoreValuesHashSet = new HashSet<>(scoreList);
            for(Double score : scoreValuesHashSet) {
                for(Entry<String, Double> innerMapEntry: termScoreMap.entrySet()) {
                    String term= innerMapEntry.getKey();
                    Double similarityScore = innerMapEntry.getValue();

                    if(score.equals(similarityScore) && sortedImportantTermsList.size() < nMostImportant) {
                        sortedImportantTermsList.add(term);
                    }
                }
            }

            nMostImportantTerms.put(docName, sortedImportantTermsList);
        }

        return nMostImportantTerms;
    }

    public static HashMap<String, HashMap<String, Double>> getTfIdf (HashMap <String,ArrayList<String>> segmentedSpeechDictionary) {
        HashMap<String, HashMap<String, Double>> tfIdfScoreMap = new HashMap<>();
        HashMap<String, Double> documentFrequencies = getDocumentFrequency(segmentedSpeechDictionary);

        for(Entry<String, ArrayList<String>> entry : segmentedSpeechDictionary.entrySet()) {
            String docName = entry.getKey();
            ArrayList<String> tokenizedTextList = entry.getValue();

            HashMap<String, Integer> tf = getTermFrequency(tokenizedTextList);
            for(Entry <String, Integer> tfEntry : tf.entrySet()) {
                String term = tfEntry.getKey();
                Integer termFreq = tfEntry.getValue();

                Double docFreq = documentFrequencies.get(term);

                double N = segmentedSpeechDictionary.size();
                double IDF = Math.log(N  / docFreq) + 1;

                double tfIdf = termFreq * IDF;

                if(tfIdfScoreMap.containsKey(docName)) {
                    tfIdfScoreMap.get(docName).put(term, tfIdf);
                } else {
                    HashMap<String,Double> newInnerMap = new HashMap<>();
                    newInnerMap.put(term,tfIdf);
                    tfIdfScoreMap.put(docName, newInnerMap);
                }
            }
        }

        return tfIdfScoreMap;
    }
    public static HashMap<String, HashMap<String, Double>> getCosineSimilarity (HashMap<String, ArrayList<String>> segmentedSpeechDictionary) {
        HashMap <String, HashMap<String, Double>> cosineSimilarity  = new HashMap<>();
        HashMap<String, HashMap<String, Double>> tfIdf = getTfIdf(segmentedSpeechDictionary);

        for(Entry <String, HashMap<String, Double>> doc1 : tfIdf.entrySet()){
            String doc1Name = doc1.getKey();
            HashMap<String, Double> doc1TfIdf = doc1.getValue(); // doc2: simil score

            for(Entry <String, HashMap<String, Double>> doc2 : tfIdf.entrySet()) {
                String doc2Name = doc2.getKey();
                HashMap<String, Double> doc2TfIdf = doc2.getValue();

                double dotProduct = 0;
                double norm1 = 0;
                double norm2 = 0;

                for( Entry<String, Double> term : doc1TfIdf.entrySet()) {
                    String termName = term.getKey();
                    double tfIdf1 = term.getValue();
                    double tfIdf2 = doc2TfIdf.getOrDefault(termName,0.0);

                    dotProduct += tfIdf1 * tfIdf2;
                    norm1 += tfIdf1 * tfIdf1;
                }

                for(double tfIdf2: doc2TfIdf.values()) {
                    norm2 += tfIdf2 * tfIdf2;
                }

                double cosineSimilarityScore = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
                if(!cosineSimilarity.containsKey(doc1Name)) {
                    cosineSimilarity.put(doc1Name,new HashMap<>());
                }
                cosineSimilarity.get(doc1Name).put(doc2Name, cosineSimilarityScore);
            }
        }
        return cosineSimilarity;
    }
}




