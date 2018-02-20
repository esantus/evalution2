package net.codejava.hibernate;

import java.sql.SQLException;
import java.util.List;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

public class SenseManager {

    public Long create(String value, SessionFactory sessionFactory) throws SQLException {
        Long id;
    	Sense sense = new Sense();
    	sense.setSense_value(value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
		String hq1 = "SELECT sense_id FROM Sense WHERE sense_value = :sense_value";
		Query<?> query = session.createQuery(hq1);
		query.setParameter("sense_value", value);
		List<?> queryResults = query.getResultList();
		if (!queryResults.isEmpty()){
			id = (Long) queryResults.get(0);
		}
		else{
		   	id = (Long) session.save(sense);
			session.getTransaction().commit();
		}
        session.close();
        
        return id; 
    }
 
    public void read(Long senseID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        
        Sense sense = session.get(Sense.class, senseID);
     
        System.out.println("Id: " + sense.getSense_id());
        System.out.println("Sense: " + sense.getSense_value());
     
        session.close();
    }
 
    public void update(Long senseID, String value, SessionFactory sessionFactory) {
        // code to modify a book
    	Sense sense = new Sense();
    	sense.setSense_id(senseID);
    	sense.setSense_value(value);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(sense);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long senseID, SessionFactory sessionFactory) {
        // code to remove a book
    	Sense sense = new Sense();
    	sense.setSense_id(senseID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(sense);
     
        session.getTransaction().commit();
        session.close();
    }
}