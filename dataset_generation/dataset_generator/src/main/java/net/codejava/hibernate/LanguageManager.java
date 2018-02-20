package net.codejava.hibernate;

import java.util.List;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

public class LanguageManager {
 
    public Long create(String value,  SessionFactory sessionFactory) {
        // code to save synsetID to DB
    	Language language = new Language();
    	language.setLanguage_value(value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
        
        String hq1 = "SELECT language_id FROM Language WHERE language_value = :language_value";
        Query<?> query = session.createQuery(hq1);
        query.setParameter("language_value", value);
        List<?> queryResults = query.getResultList();
        Long id;
        if (!queryResults.isEmpty()){
        	id = (Long) queryResults.get(0);
        }
        else{
           	id = (Long) session.save(language);
        	session.getTransaction().commit();
        }
        session.close();
       
        return id;
    }
    
    public void read(Long languageID,  SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        
        Language language = session.get(Language.class, languageID);
     
        System.out.println("Id: " + language.getLanguage_id());
        System.out.println("Language: " + language.getLanguage_value());
     
        session.close();
    }
 
    public void update(Long languageID, String value,  SessionFactory sessionFactory) {
        // code to modify a book
    	Language language = new Language();
    	language.setLanguage_id(languageID);
    	language.setLanguage_value(value);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(language);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long languageID,  SessionFactory sessionFactory) {
        // code to remove a book
    	Language language = new Language();
    	language.setLanguage_id(languageID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(language);
     
        session.getTransaction().commit();
        session.close();
    }
}