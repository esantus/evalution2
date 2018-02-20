package net.codejava.hibernate;

import org.hibernate.Session;
import org.hibernate.SessionFactory;

public class AllWordSensesManager {
 
	public void create(Long wordID, Long languageID, Long wordSenseID, Long sourceID, SessionFactory sessionFactory) {
        AllWordSensesID allSensesID = new AllWordSensesID();
    	AllWordSenses allSenses = new AllWordSenses();
    	
    	allSensesID.setWord_id(wordID);
    	allSensesID.setLanguage_id(languageID);
    	allSensesID.setWordSenses_id(wordSenseID);
    	
    	allSenses.setId(allSensesID);
    	allSenses.setSource_id(sourceID);
    	
    	Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        if(session.get(AllWordSenses.class, allSensesID) == null){
        	session.save(allSenses);
        }
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void read(Long wordSenseID, Long wordID, Long languageID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        AllWordSensesID allSensesID = new AllWordSensesID();
        allSensesID.setWordSenses_id(wordSenseID);
        allSensesID.setWord_id(wordID);
        allSensesID.setLanguage_id(languageID);
        
        AllWordSenses w2syn = session.get(AllWordSenses.class, allSensesID);
     
        System.out.println("Id: " + w2syn.getId());
        System.out.println("Sense: " + w2syn.getSource_id());
        
        session.close();
    }
 
    public void update(Long wordSenseID, Long wordID, Long languageID, Long sourceID, SessionFactory sessionFactory) {
        // code to modify a book
        AllWordSensesID allSensesID = new AllWordSensesID();
    	AllWordSenses allSenses = new AllWordSenses();
    	
    	allSensesID.setWordSenses_id(wordSenseID);
    	allSensesID.setWord_id(wordID);
    	allSensesID.setLanguage_id(languageID);
    	
    	allSenses.setId(allSensesID);
    	allSenses.setSource_id(sourceID);
    	   
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(allSenses);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long wordSenseID, Long wordID, Long languageID, SessionFactory sessionFactory) {
        // code to remove a book
        AllWordSensesID allSensesID = new AllWordSensesID();
    	AllWordSenses allSenses = new AllWordSenses();

    	allSensesID.setWordSenses_id(wordSenseID);
    	allSensesID.setWord_id(wordID);
    	allSensesID.setLanguage_id(languageID);
    	
    	allSenses.setId(allSensesID);
        
    	Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(allSenses);
     
        session.getTransaction().commit();
        session.close();
    }
}