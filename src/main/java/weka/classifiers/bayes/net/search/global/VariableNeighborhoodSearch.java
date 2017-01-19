package weka.classifiers.bayes.net.search.global;

import java.util.Collections;
import java.util.Vector;
import java.util.concurrent.ThreadLocalRandom;

import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


public class VariableNeighborhoodSearch extends HillClimber implements TechnicalInformationHandler  {
	
	/** for serialization */
	private static final long serialVersionUID = 1L;
	
	/** number of Neighborhoods **/
	int m_nNeighborhoods = 3;
	/** number of Iterations **/
	int stopCriteria=3;

	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

	    result = new TechnicalInformation(Type.TECHREPORT);
	    result.setValue(Field.AUTHOR, "Emir Causevic, Nan Li, Wang Di");
	    result.setValue(Field.YEAR, "2017");
	    result.setValue(Field.TITLE,
	      "Bayesian Belief Networks: VariableNeighbourhoodSearch");
	    result.setValue(Field.INSTITUTION, "TU Vienna");
	    result.setValue(Field.ADDRESS, "TU Vienna");

	    return result;
	}
	 /**
	   * search determines the network structure/graph of the network with the 
	   * Variable Neighborhood Search algorithm.
	   * 
	   * @param bayesNet the network to use
	   * @param instances the instances to use
	   * @throws Exception if something goes wrong
	   */
	  @Override
	  protected void search(BayesNet bayesNet, Instances instances) throws Exception {
		  Operation operation = new Operation();
		  // keeps track of score: best structure found so far
		  double fBestScore;
		  double fCurrentScore = calcScore(bayesNet);
		  int iRun=0, nRun=0, randNeighborhood;
		  

		  // keeps track of best structure found so far
		  BayesNet bestBayesNet;

		  // initialize bestBayesNet
		  fBestScore = fCurrentScore;
		  bestBayesNet = new BayesNet();
		  bestBayesNet.m_Instances = instances;
		  bestBayesNet.initStructure();
		  copyParentSets(bestBayesNet, bayesNet);
		  
		  // generate initial solution
	      fCurrentScore =  calcScore(bayesNet);
	      fBestScore = fCurrentScore;
	      // perform VNS
	      while(iRun <= stopCriteria){
	    	  while(nRun <= m_nNeighborhoods){
	    		  randNeighborhood = ThreadLocalRandom.current().nextInt(0, m_nNeighborhoods);
	    		  switch (randNeighborhood){
	    			  case 0:
	    				 operation = findBestArcToAdd(bayesNet, instances, operation);
	    				 performOperation(bayesNet, instances, operation);
	    				  break;
	    			  case 1:
	    				  operation = findBestArcToDelete(bayesNet, instances, operation);
	    				  performOperation(bayesNet, instances, operation);
	    				  break;
	    			  case 2:
	    				  operation = findBestArcToReverse(bayesNet, instances, operation);
	    				  performOperation(bayesNet, instances, operation);
	    				  break;
	    		  }
			  
			  // check if better solution has been found and advance VNS
	    		  fCurrentScore = operation.m_fScore;
	    		  if (fCurrentScore > fBestScore) {
	    			  fBestScore = fCurrentScore;
	    			  copyParentSets(bestBayesNet, bayesNet);
	    			  nRun=0;
	    		  }else{
	    			  nRun++;
			 	}
	    	  }
	    	  iRun++;
	    	  nRun=0;
	      }
	      // restore current network to best network
	      copyParentSets(bayesNet, bestBayesNet);

	      // free up memory
	      bestBayesNet = null;
	  }
	  
	  /**
	   * copyParentSets copies parent sets of source to dest BayesNet
	   * 
	   * @param dest destination network
	   * @param source source network
	   */
	  void copyParentSets(BayesNet dest, BayesNet source) {
	    int nNodes = source.getNrOfNodes();
	    // clear parent set first
	    for (int iNode = 0; iNode < nNodes; iNode++) {
	      dest.getParentSet(iNode).copy(source.getParentSet(iNode));
	    }
	  } // CopyParentSets
	  
	  /**
	   * @return Get stopCriteria
	   */
	  public int getstopCriteria() {
	    return stopCriteria;
	  } // getstopCriteria

	  /**
	   * Sets the stopCriteria
	   * 
	   * @param stopCriteria The number of runs to set
	   */
	  public void setstopCriterian(int stopCriteria) {
		  this.stopCriteria = stopCriteria;
	  } // setstopCriterian
	  
	  
	  @Override
	  public void setOptions(String[] options) throws Exception {
	    String stopCriteriaOp = Utils.getOption('S', options);
	    if (stopCriteriaOp.length() != 0) {
	    	setstopCriterian(Integer.parseInt(stopCriteriaOp));
	    }
	    super.setOptions(options);
	  } // setOptions

	  /**
	   * Gets the current settings of the search algorithm.
	   * 
	   * @return an array of strings suitable for passing to setOptions
	   */
	  @Override
	  public String[] getOptions() {

	    Vector<String> options = new Vector<String>();
	    options.add("-S");
	    options.add("" + getstopCriteria());

	    Collections.addAll(options, super.getOptions());

	    return options.toArray(new String[0]);
	  } // getOptions

	  /**
	   * This will return a string describing the classifier.
	   * 
	   * @return The string.
	   */
	  @Override
	  public String globalInfo() {
	    return "This Bayes Network learning algorithm uses VNS for finding a well scoring "
	      + "Bayes network structure. VNS uses 3 basic neighborhoods: add arc, delete arc "
	      + "and reverse arc. The neighborhood is randomlly chosen and local search is done."
	      + "If better result is found the procedure is repeated and counter is 0."
	      + "Otherwise it is increased and procedure is done certian number of iterations..\n\n"
	      + "For more information see:\n\n" + getTechnicalInformation().toString();
	  } // globalInfo
	  
	  
}
