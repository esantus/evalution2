package net.codejava.hibernate;

import java.util.List;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

public class SynsetDomainManager {
 
    public Long create(String value, SessionFactory sessionFactory) {
        // code to save synsetID to DB
    	SynsetDomain domain = new SynsetDomain();
    	domain.setSynsetDomain_value(value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        String hq1 = "SELECT synsetDomain_id FROM SynsetDomain WHERE synsetDomain_value = :synsetDomain_value";
        Query<?> query = session.createQuery(hq1);
        query.setParameter("synsetDomain_value", value);
        List<?> queryResults = query.getResultList();
        Long id;
        if (!queryResults.isEmpty()){
        	id = (Long) queryResults.get(0);
        }
        else{
           	id = (Long) session.save(domain);
        	session.getTransaction().commit();
        }
        session.close();
        return id;
    }
 
    public void read(Long domainID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        
        SynsetDomain domain = session.get(SynsetDomain.class, domainID);
     
        System.out.println("Id: " + domain.getSynsetDomain_id());
        System.out.println("Domain: " + domain.getSynsetDomain_value());
     
        session.close();
    }
 
    public void update(Long domainID, String value, SessionFactory sessionFactory) {
        // code to modify a book
    	SynsetDomain domain = new SynsetDomain();
    	domain.setSynsetDomain_id(domainID);
    	domain.setSynsetDomain_value(value);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(domain);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long domainID, SessionFactory sessionFactory) {
    	SynsetDomain domain = new SynsetDomain();
    	domain.setSynsetDomain_id(domainID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(domain);
     
        session.getTransaction().commit();
        session.close();
    }
}