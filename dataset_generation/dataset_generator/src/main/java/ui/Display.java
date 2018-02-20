package ui;

import compare.ResourceComparison;
import configuration.Configuration;


public class Display { 

	public static void main(String[] args){
		
	Configuration.loadPropertiesFile();
	
	System.out.println("Welcome to the BabelNet extractor!"); 
	
	ResourceComparison comparer = new ResourceComparison();
	comparer.resourceAligner();
	
	}	
}
