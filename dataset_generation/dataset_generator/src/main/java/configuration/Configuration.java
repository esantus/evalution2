/**
* Class to load the configuration file of the application rdfClass.
* @author: Dagmar Gromann
* @version: 1.0*/

package configuration;

import java.util.Properties;
import java.io.*;

public class Configuration {
	
	//Variable of type Properties references a hash map.
	public static Properties dataSource = new Properties();
	public static BufferedInputStream stream;
	
	/**
	 * This method serves to load the basic properties-file containing
	 * the input sources and paths for this application.
	 */
	public static void loadPropertiesFile(){
		try {
			stream = new BufferedInputStream(new FileInputStream("BabelNetHibernate.properties"));
		} catch(FileNotFoundException e){
			System.out.println("Properties file could not be found!");
		}
		try {
			dataSource.load(stream);
		} catch (IOException e) {
			System.out.println("Properties-File could not be loaded");
		}	
	}
}
