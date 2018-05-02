package net.codejava.hibernate;

import org.hibernate.Session;
import org.hibernate.SessionFactory;

public class SynsetRelationsManager {
 
    public void create(Long sourceSynset_id, Long targetSynset_id, Long relation_id, Double relationWeight, String relationSurfaceText, SessionFactory sessionFactory) {
    	SynsetRelationsID srID = new SynsetRelationsID();
    	SynsetRelations synrel = new SynsetRelations();
    	
    	srID.setSourceSynset_id(sourceSynset_id);
    	srID.setTargetSynset_id(targetSynset_id);
    	srID.setRelation_id(relation_id);
    	synrel.setId(srID);
    	synrel.setRelationSurfaceText(relationSurfaceText);
    	synrel.setRelationWeight(relationWeight);
    	
    	Session session = sessionFactory.openSession();
        session.beginTransaction();
        if(session.get(SynsetRelations.class, srID) == null){
        	session.save(synrel);
        }
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void read(Long sourceSynset_id, Long targetSynset_id, Long relation_id, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        SynsetRelationsID srID = new SynsetRelationsID();
        srID.setSourceSynset_id(sourceSynset_id);
        srID.setTargetSynset_id(targetSynset_id);
        srID.setRelation_id(relation_id);
        
        SynsetRelations synrel = session.get(SynsetRelations.class, srID);
     
        System.out.println("Id: " + synrel.getId());
        System.out.println("Sense: " + synrel.getRelationWeight());
        System.out.println("Text: "+ synrel.getRelationSurfaceText());
        
        session.close();
    }
 
    public void update(Long sourceSynset_id, Long targetSynset_id, Long relation_id, Double relationWeight, String relationSurfaceText, SessionFactory sessionFactory){
    	SynsetRelationsID srID = new SynsetRelationsID();
    	SynsetRelations synrel = new SynsetRelations();
    	
    	srID.setSourceSynset_id(sourceSynset_id);
    	srID.setTargetSynset_id(targetSynset_id);
    	srID.setRelation_id(relation_id);
    	synrel.setId(srID);
    	synrel.setRelationSurfaceText(relationSurfaceText);
    	synrel.setRelationWeight(relationWeight);
    	
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(synrel);
     
        session.getTransaction().commit();
        session.close();
    }
    
    public void delete(Long sourceSynset_id, Long targetSynset_id, Long relation_id, SessionFactory sessionFactory) {
    	SynsetRelationsID srID = new SynsetRelationsID();
    	SynsetRelations synrel = new SynsetRelations();
    	
    	srID.setSourceSynset_id(sourceSynset_id);
    	srID.setTargetSynset_id(targetSynset_id);
    	srID.setRelation_id(relation_id);
    	synrel.setId(srID);
     
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(synrel);
     
        session.getTransaction().commit();
        session.close();
    }
}