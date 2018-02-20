package net.codejava.hibernate;
import javax.persistence.*;

@Entity
@Table(name="synsetDomain")
public class SynsetDomain {
	private Long synsetDomain_id;
	private String synsetDomain_value;
	
	public SynsetDomain() {
		super();
		// TODO Auto-generated constructor stub
	}
	public SynsetDomain(Long synsetDomain_id, String synsetDomain_value) {
		super();
		this.synsetDomain_id = synsetDomain_id;
		this.synsetDomain_value = synsetDomain_value;
	}
	
	@Id
	@GeneratedValue(strategy=GenerationType.IDENTITY)
	@Column(name = "synsetDomain_id")
	public Long getSynsetDomain_id() {
		return synsetDomain_id;
	}
	public void setSynsetDomain_id(Long synsetDomain_id) {
		this.synsetDomain_id = synsetDomain_id;
	}
	public String getSynsetDomain_value() {
		return synsetDomain_value;
	}
	public void setSynsetDomain_value(String synsetDomain_value) {
		this.synsetDomain_value = synsetDomain_value;
	}
	
	public String toString() {
		return "Domain [synsetDomain_id=" + synsetDomain_id + ", synsetDomain_value=" + synsetDomain_value + "]";
	}
	
}
