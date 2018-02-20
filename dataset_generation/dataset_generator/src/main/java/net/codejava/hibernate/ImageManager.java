package net.codejava.hibernate;

import java.util.List;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

public class ImageManager {
 
    public Long create(String value, SessionFactory sessionFactory) {
        // code to save synsetID to DB
    	Image image = new Image();
    	image.setImage_value(value);
    	
        Session session = sessionFactory.openSession();
        session.beginTransaction();
        
        String hq1 = "SELECT image_id FROM Image WHERE image_value = :image_value";
        Query<?> query = session.createQuery(hq1);
        query.setParameter("image_value", value);
        List<?> queryResults = query.getResultList();
        Long id;
        if (!queryResults.isEmpty()){
        	id = (Long) queryResults.get(0);
        }
        else{
           	id = (Long) session.save(image);
        	session.getTransaction().commit();
        }
        session.close();
        return id;
    }
 
    public void read(Long imageID, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        
        Image image = session.get(Image.class, imageID);
     
        System.out.println("Id: " + image.getImage_id());
        System.out.println("Image: " + image.getImage_value());
     
        session.close();
    }
 
    public void update(Long imageID, String value, SessionFactory sessionFactory) {
        // code to modify a book
    	Image image = new Image();
    	image.setImage_id(imageID);
    	image.setImage_value(value);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(image);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long imageID, SessionFactory sessionFactory) {
        // code to remove a book
    	Image image = new Image();
    	image.setImage_id(imageID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(image);
     
        session.getTransaction().commit();
        session.close();
    }
}