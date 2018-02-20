package net.codejava.hibernate;

import java.util.List;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

public class RelationNameManager {
 
    public Long create(String value, SessionFactory sessionFactory) {
        // code to save synsetID to DB
    	RelationName rel = new RelationName();
    	rel.setRelationName_value(value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
        String hq1 = "SELECT relationName_id FROM RelationName WHERE relationName_value = :relationName_value";
        Query<?> query = session.createQuery(hq1);
        query.setParameter("relationName_value", value);
        List<?> queryResults = query.getResultList();
        Long id;
        if (!queryResults.isEmpty()){
        	id = (Long) queryResults.get(0);
        }
        else{
           	id = (Long) session.save(rel);
        	session.getTransaction().commit();
        }
        session.close();
        
        return id; 
    }
 
    public void read(Long relationName_id, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        
        RelationName rel = session.get(RelationName.class, relationName_id);
     
        System.out.println("Id: " + rel.getRelationName_id());
        System.out.println("RelationName: " + rel.getRelationName_value());
     
        session.close();
    }
 
    public void update(Long relationName_id, String relationName_value, SessionFactory sessionFactory) {
        // code to modify a book
    	RelationName rel = new RelationName();
    	rel.setRelationName_id(relationName_id);
    	rel.setRelationName_value(relationName_value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(rel);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long relationName_id, SessionFactory sessionFactory) {
        // code to remove a book
    	RelationName rel = new RelationName();
    	rel.setRelationName_id(relationName_id);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(rel);
     
        session.getTransaction().commit();
        session.close();
    }
}