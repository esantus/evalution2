package net.codejava.hibernate;

import java.util.List;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

public class DefinitionManager {

    public Long create(String value, SessionFactory sessionFactory) {
        // code to save synsetID to DB
    	Definition def = new Definition();
    	def.setDefinition_value(value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
        
        String hq1 = "SELECT definition_id FROM Definition WHERE definition_value = :definition_value";
        Query<?> query = session.createQuery(hq1);
        query.setParameter("definition_value", value);
        List<?> queryResults = query.getResultList();
        Long id;
        if (!queryResults.isEmpty()){
        	id = (Long) queryResults.get(0);
        }
        else{
           	id = (Long) session.save(def);
        	session.getTransaction().commit();
        }
        session.close();
        
        return id; 
    }
 
    public void read(Long defID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        
        Definition def = session.get(Definition.class, defID);
     
        System.out.println("Id: " + def.getDefinition_id());
        System.out.println("Definition: " + def.getDefinition_value());
     
        session.close();
    }
 
    public void update(Long defID, String value, SessionFactory sessionFactory) {
        // code to modify a book
    	Definition def = new Definition();
    	def.setDefinition_id(defID);
    	def.setDefinition_value(value);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(def);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long defID, SessionFactory sessionFactory) {
        // code to remove a book
    	Definition def = new Definition();
    	def.setDefinition_id(defID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(def);
     
        session.getTransaction().commit();
        session.close();
    }
}